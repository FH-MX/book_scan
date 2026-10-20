# scan2pdf.py — 本の写真（JPEG）をページごとに整形して、文字検索できる PDF にする（試作 v0.4）
import argparse  # コマンドの引数を読むため
import re  # テキストの空白を整えるため
import subprocess  # ocrmypdf を呼び出すため
from pathlib import Path  # パスを扱うため

import cv2  # 画像処理
import numpy as np  # 配列計算
from PIL import Image, ImageOps  # EXIF の回転の適用と PDF の書き出し

DEPTH_MIN = 40.0  # これより浅い谷は、綴じ目の検出の信頼度が低いとみなす


def load(path):  # EXIF の回転を適用して、OpenCV の BGR 配列で返す
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # 回転を反映して RGB にする
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # OpenCV の並びに直す


def find_spine(gray):  # 綴じ目を、周りより暗い細い縦の谷として探す
    h, w = gray.shape  # 画像の高さと幅
    lo, hi = int(w * 0.30), int(w * 0.70)  # 探す範囲は幅の 30〜70%
    band = gray[int(h * 0.15):int(h * 0.85), :].astype(np.float32)  # 上下の端は背景なので除く
    profile = band.mean(axis=0)  # 列ごとの平均の明るさ
    k = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2)  # ガウス型の平滑化カーネル
    k /= k.sum()  # 合計を 1 にそろえる
    profile = np.convolve(profile, k, mode="same")  # 細かい揺れをならす
    best_x, best_v = lo, -1e9  # 最良の位置と谷の深さ
    for x in range(lo, hi):  # 候補の位置ごとに谷の深さを測る
        side = (profile[x - 250:x - 100].mean() + profile[x + 100:x + 250].mean()) / 2  # 両側の平均の明るさ
        v = side - profile[x]  # 両側より、その位置がどれだけ暗いか
        if v > best_v:  # より深ければ更新する
            best_x, best_v = x, v  # 位置と深さを覚える
    return best_x, best_v  # 位置と深さを返す


def find_gap(img):  # 左右のページの文字の間にある、文字のない縦の帯の中央を探す（谷が浅いときの代わり）
    gray = cv2.cvtColor(flatten_light(img), cv2.COLOR_BGR2GRAY)  # 照明のむらを消した灰色
    h, w = gray.shape  # 高さと幅
    ink = (gray[int(h * 0.2):int(h * 0.8), :] < 128).mean(axis=0)  # 列ごとの暗い画素（文字）の割合
    ink = np.convolve(ink, np.ones(31) / 31, mode="same")  # 細かい揺れをならす
    lo, hi = int(w * 0.35), int(w * 0.65)  # 探す範囲は幅の 35〜65%
    quiet = ink[lo:hi] < 0.01  # 文字がほとんどない列
    best_start, best_len, start = None, 0, None  # 一番長い区間の始まりと長さ、いま数えている区間の始まり
    for i, q in enumerate(list(quiet) + [False]):  # 末尾に False を足して、最後の区間も閉じる
        if q and start is None:  # 区間が始まったら
            start = i  # 始まりを覚える
        elif not q and start is not None:  # 区間が終わったら
            if i - start > best_len:  # これまでより長ければ
                best_start, best_len = start, i - start  # 覚える
            start = None  # 数え直す
    if best_start is None:  # 文字のない帯がなければ
        return None, 0  # 見つからない
    return lo + best_start + best_len // 2, best_len  # 帯の中央と、帯の幅を返す


def flatten_light(img):  # 背景の明るさで割って、紙を白く・照明のむらを消す（色は保つ）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 背景の推定用の灰色画像
    small = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)  # 縮小して背景を推定しやすくする
    bg = cv2.medianBlur(small, 31)  # 文字を消した背景の明るさ
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)  # 元の大きさに戻す
    bg = np.maximum(bg, 1)  # 0 で割らないようにする
    out = np.empty_like(img)  # 出力の入れ物
    for c in range(3):  # 色ごとに同じ背景で割る
        out[:, :, c] = cv2.divide(np.ascontiguousarray(img[:, :, c]), bg, scale=255)  # 背景で割る
    return out  # 色を保ったまま、紙が白くなった画像


def skew_angle(gray):  # 行の投影のばらつきが最大になる傾きを探す（-6〜6 度）
    h, w = gray.shape  # 高さと幅
    core = gray[int(h * 0.15):int(h * 0.85), int(w * 0.05):int(w * 0.80)]  # 紙の中央付近だけを使う
    if (core < 100).mean() < 0.003:  # 暗い画素（文字）がほとんどなければ
        return 0.0  # 白紙とみなして回さない
    ink = 255 - cv2.resize(core, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)  # 縮小して、文字を白・紙を黒にする
    sh, sw = ink.shape  # 縮小後の大きさ
    best_a, best_v, v0 = 0.0, -1.0, 0.0  # 最良の角度・そのばらつき・傾き 0 度のばらつき
    for a in np.arange(-6.0, 6.01, 0.5):  # 角度を 0.5 度刻みで試す
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(a), 1.0)  # 回転行列
        r = cv2.warpAffine(ink, m, (sw, sh), flags=cv2.INTER_LINEAR, borderValue=0)  # 回してみる
        v = float(r.sum(axis=1).astype(np.float64).var())  # 行ごとの合計のばらつき
        if abs(a) < 1e-9:  # 傾き 0 度のとき
            v0 = v  # 基準として覚える
        if v > best_v:  # より大きければ更新する
            best_a, best_v = float(a), v  # 角度とばらつきを覚える
    return best_a if best_v > 1.1 * v0 and abs(best_a) <= 3.0 else 0.0  # 差が小さいときや 3 度を超える異常な値のときは回さない


def rotate(img, a):  # 画像を a 度回す（空いた所は白で埋める）
    if a == 0.0:  # 回さないときは
        return img  # そのまま返す
    h, w = img.shape[:2]  # 高さと幅
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)  # 回転行列
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))  # 回して返す


def shrink(img, max_h):  # 高さが max_h を超えるときだけ縮小する
    h, w = img.shape[:2]  # 高さと幅
    if h <= max_h:  # すでに小さければ
        return img  # そのまま返す
    return cv2.resize(img, (round(w * max_h / h), max_h), interpolation=cv2.INTER_AREA)  # 縦横比を保って縮小


def order_points(pts):  # 4 点を 左上・右上・右下・左下 の順に並べる
    s = pts.sum(axis=1)  # x + y
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # 左上・右上・右下・左下


def warp_largest(img, mask, f):  # 縮小画像のマスクで一番大きい領域を 4 角形とみなし、長方形に補正して切り出す
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # 文字などの穴を埋める
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))  # 細い突起を消す
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)  # つながった領域に分ける
    if n < 2:  # 領域が見つからなければ
        return img  # そのまま返す
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # 一番大きい領域
    comp = (labels == k).astype(np.uint8) * 255  # その領域だけの画像
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭を取る
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))  # 凸包にする
    box = None  # 4 つの角
    for eps in (0.01, 0.02, 0.03, 0.05):  # 近似の粗さを変えて 4 角形を探す
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)  # 多角形に近似する
        if len(approx) == 4:  # 4 角形になったら
            box = approx.reshape(4, 2).astype(np.float32)  # 角を取り出す
            break  # 探索をやめる
    if box is None:  # 4 角形にならなければ
        box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)  # 回転した長方形で代用する
    box = order_points(box) / f  # 元の大きさの座標にして並べる
    center = box.mean(axis=0)  # 4 角形の中心
    box = center + (box - center) * 1.015  # 端を切りすぎないよう、外側へ 1.5% 広げる
    wid = int(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3])))  # 出力の幅
    hei = int(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1])))  # 出力の高さ
    if wid * hei < 0.3 * img.shape[0] * img.shape[1]:  # 小さすぎるときは検出の失敗とみなす
        return img  # そのまま返す
    dst = np.array([[0, 0], [wid - 1, 0], [wid - 1, hei - 1], [0, hei - 1]], dtype=np.float32)  # 出力の 4 隅
    m = cv2.getPerspectiveTransform(box, dst)  # 台形から長方形への変換
    return cv2.warpPerspective(img, m, (wid, hei), flags=cv2.INTER_CUBIC)  # 変換して返す


def crop_page(img):  # 本文のページ：彩度が低く暗すぎない所を紙とみなして切り出す
    f = 0.25  # 検出は縮小した画像で行う
    hsv = cv2.cvtColor(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)  # 縮小して色相・彩度・明度にする
    mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 50)).astype(np.uint8) * 255  # 綴じ目の影も含めて紙とみなす
    return warp_largest(img, mask, f)  # 一番大きい紙の領域を切り出す


def crop_cover(img):  # 表紙：写真の縁の色相（机など）と違う所を本とみなして切り出す
    f = 0.25  # 検出は縮小した画像で行う
    hsv = cv2.cvtColor(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV).astype(np.int16)  # 縮小して色相・彩度・明度にする
    b = max(2, int(min(hsv.shape[:2]) * 0.02))  # 縁として使う幅
    border = np.concatenate([hsv[:b].reshape(-1, 3), hsv[-b:].reshape(-1, 3), hsv[:, :b].reshape(-1, 3), hsv[:, -b:].reshape(-1, 3)])  # 上下左右の縁の画素
    ref_h = int(np.median(border[:, 0]))  # 背景の代表の色相（0〜179）
    dh = np.abs(hsv[:, :, 0] - ref_h)  # 色相の差
    dh = np.minimum(dh, 180 - dh)  # 色相は一周するので、近い方の差にする
    colored = (dh > 12) & (hsv[:, :, 1] > 60)  # 背景と色相が違い、色のある所
    whitish = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 120)  # 白っぽい所（帯やラベル）
    mask = (colored | whitish).astype(np.uint8) * 255  # どちらかなら本とみなす
    return warp_largest(img, mask, f)  # 一番大きい本の領域を切り出す


def best_shift(a, b, win):  # 帯 a に対して、帯 b の行が何 px 下にずれているかを、相関が最大になるずれで求める
    norm = np.sqrt(float((a * a).sum()) * float((b * b).sum()))  # 相関を 0〜1 にそろえるための大きさ
    if norm == 0:  # 文字がない帯なら
        return 0  # ずれなしとする
    best, best_d = -1e18, 0  # 最大の相関とそのずれ
    for d in range(-win, win + 1):  # ずれを 1px ずつ試す
        s = float((a[:len(a) - d] * b[d:]).sum()) if d >= 0 else float((a[-d:] * b[:len(b) + d]).sum())  # ずらして掛け合わせた合計
        if s > best:  # より大きければ
            best, best_d = s, d  # 覚える
    return best_d if best / norm >= 0.3 else 0  # 相関が弱いときは信用しない


def dewarp_by_lines(img):  # 文字の行の上下のずれを列ごとに測り、行がまっすぐ並ぶように縦へずらす
    f = 0.25  # 測定は縮小した画像で行う
    gray = cv2.cvtColor(cv2.resize(flatten_light(img), None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)  # むらを消して縮小した灰色
    ink = np.clip(200.0 - gray, 0, None)  # 文字の濃さ（紙は 0）
    h, w = ink.shape  # 縮小後の高さと幅
    ref = ink.sum(axis=1) - ink.sum(axis=1).mean()  # ページ全体の行ごとの濃さ（平均を引く）
    ac = np.correlate(ref, ref, mode="full")[len(ref) - 1:]  # 自己相関（行の繰り返しの間隔を見るため）
    if len(ac) < 60:  # 画像が小さすぎれば
        return img  # 直さない
    lim = len(ac) // 4  # 探す範囲の上限
    first_min = next((i for i in range(1, lim - 1) if ac[i] < ac[i - 1] and ac[i] <= ac[i + 1]), None)  # 自己相関の最初の谷
    if first_min is None:  # 谷がなければ（行の繰り返しがない）
        return img  # 直さない
    pitch = first_min + int(np.argmax(ac[first_min:lim]))  # 谷の後の一番高い山＝行の間隔
    win = max(2, int(pitch * 0.4))  # 隣の帯どうしで探すずれの範囲（1 行分ずれて誤らないよう、間隔の 4 割まで）
    nx, ny = 24, 4  # 横に 24、縦に 4 つの区画に分ける
    xs = np.linspace(0, w, nx + 1).astype(int)  # 区画の横の境目
    ys = np.linspace(0, h, ny + 1).astype(int)  # 区画の縦の境目
    D = np.zeros((ny, nx))  # 区画ごとのずれ（中央の帯が基準）
    for j in range(ny):  # 縦の区画ごとに
        band = ink[ys[j]:ys[j + 1]]  # その高さの範囲
        prof = [band[:, xs[i]:xs[i + 1]].sum(axis=1) for i in range(nx)]  # 帯ごとの、行ごとの濃さ
        prof = [p - p.mean() for p in prof]  # 平均を引く
        for step in (1, -1):  # 中央から右へ、左へ
            i = nx // 2  # 中央の帯から始める
            while 0 <= i + step < nx:  # 端まで
                D[j, i + step] = D[j, i] + best_shift(prof[i], prof[i + step], win)  # 隣とのずれを足していく
                i += step  # 次の帯へ
    cx = (xs[:-1] + xs[1:]) / 2.0  # 帯の中心の x
    for j in range(ny):  # 縦の区画ごとに
        D[j] = np.polyval(np.polyfit(cx, D[j], 2), cx)  # 2 次式でならして、ばらつきを抑える
    print(f"  行の曲がり：行間隔 {pitch / f:.0f}px、最大のずれ {np.abs(D).max() / f:.0f}px")  # 確認用に補正量を表示する
    if np.abs(D).max() < 1.0:  # ずれが小さければ（縮小後で 1px 未満）
        return img  # 直さない
    H, W = img.shape[:2]  # 元の高さと幅
    off = cv2.resize((D / f).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)  # 元の大きさでのずれ
    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))  # 元の座標
    return cv2.remap(img, map_x, map_y + off, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  # 縦にずらして返す


def prepare_text_page(img, max_h):  # 本文のページを整える（紙を切り出し、行の曲がりを戻し、明るさのむらを消し、傾きを直し、縮小する）
    flat = flatten_light(dewarp_by_lines(crop_page(img)))  # 紙を切り出し、行の曲がりを戻してから、明るさのむらを消す
    ang = skew_angle(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY))  # 傾きを求める
    return shrink(rotate(flat, ang), max_h), ang  # 傾きを直して縮小し、角度も返す


def decide_cut(img, name, cuts):  # 見開きの切り位置を決めて、説明と一緒に返す
    if name in cuts:  # 手動の指定があれば
        return int(cuts[name]), "手動の指定"  # それを使う
    x, depth = find_spine(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # 綴じ目の谷を探す
    if depth >= DEPTH_MIN:  # 谷が十分に深ければ
        return x, f"綴じ目の谷（深さ {depth:.1f}）"  # それを使う
    gx, glen = find_gap(img)  # 谷が浅いので、文字のない帯を探す
    if gx is not None and glen >= 60:  # 十分な幅の帯があれば
        return gx, f"谷が浅い（{depth:.1f}）ので文字のない帯の中央（幅 {glen}px）"  # それを使う
    return x, f"警告：谷が浅く（{depth:.1f}）文字のない帯も見つからない。--cut で指定を"  # 仕方なく谷の位置を使う


def main():  # 全体の処理
    ap = argparse.ArgumentParser()  # 引数の定義
    ap.add_argument("inputs", nargs="+", help="写真のファイル（撮影順）")  # 入力の写真
    ap.add_argument("-o", "--output", required=True, help="出力する PDF")  # 出力の PDF
    ap.add_argument("--rtl", action="store_true", help="右開き（縦書きの本）。右のページが先")  # 綴じ方向
    ap.add_argument("--max-height", type=int, default=2600, help="ページ画像の高さの上限（px）")  # 縮小の上限
    ap.add_argument("--quality", type=int, default=80, help="JPEG の品質")  # 画質
    ap.add_argument("--dpi", type=float, default=250, help="PDF の解像度の設定値")  # ページの大きさに効く
    ap.add_argument("--cut", action="append", default=[], help="綴じ目を手で指定する（例：IMG_0774.JPG=2750）")  # 手動の切り位置
    ap.add_argument("--no-ocr", action="store_true", help="OCR をしない")  # OCR を省く
    args = ap.parse_args()  # 引数を読む
    cuts = dict(c.split("=") for c in args.cut)  # 手動の切り位置を辞書にする
    pages = []  # 出力するページを順に入れる
    for path in map(Path, args.inputs):  # 写真を 1 枚ずつ処理する
        img = load(path)  # 写真を読む
        h, w = img.shape[:2]  # 回転を反映した後の高さと幅
        if h >= w:  # 縦長なら、表紙などの単ページ
            pages.append(shrink(crop_cover(img), args.max_height))  # 本の部分を切り出して、色のまま縮小する
            print(f"{path.name}: 単ページ（縦長）")  # 判定を表示する
            continue  # 次の写真へ
        x, how = decide_cut(img, path.name, cuts)  # 切り位置を決める
        halves = [img[:, :x], img[:, x:]]  # 左右に割る
        if args.rtl:  # 右開きなら
            halves.reverse()  # 右のページを先にする
        angs = []  # 左右の傾き
        for half in halves:  # 左右のページを順に整える
            page, ang = prepare_text_page(half, args.max_height)  # 整える
            pages.append(page)  # ページに加える
            angs.append(ang)  # 傾きを覚える
        print(f"{path.name}: 見開き 切り位置 {x}px（{how}）傾き {angs[0]} / {angs[1]} 度")  # 結果を表示する
    out = Path(args.output)  # 出力先
    tmp = out.with_suffix(".noocr.pdf")  # OCR の前の PDF
    images = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]  # PIL の画像にする
    images[0].save(tmp, "PDF", save_all=True, append_images=images[1:], resolution=args.dpi, quality=args.quality)  # PDF に書き出す
    print(f"ページ数 {len(pages)}、OCR 前の PDF: {tmp}")  # 進み具合を表示する
    if args.no_ocr:  # OCR を省くなら
        return  # ここで終わる
    side = out.with_suffix(".raw.txt")  # OCR の生のテキスト
    subprocess.run(["ocrmypdf", "-l", "jpn", "--output-type", "pdf", "--optimize", "1", "--sidecar", str(side), str(tmp), str(out)], check=True)  # OCR をかけ、テキストも書き出す
    text = side.read_text(encoding="utf-8")  # 生のテキストを読む
    jp = r"[　-ヿ一-鿿＀-￯]"  # 日本語の文字（記号・かな・漢字・全角）
    text = re.sub(rf"(?<={jp}) +(?={jp})", "", text)  # 日本語の文字の間に入った空白を消す
    out.with_suffix(".txt").write_text(text, encoding="utf-8")  # AI に読ませる用のテキストとして保存する
    side.unlink()  # 生のテキストは消す
    print(f"完成: {out}（テキスト: {out.with_suffix('.txt')}）")  # 完成を表示する


if __name__ == "__main__":  # 直接実行されたときだけ
    main()  # 実行する
