import cv2

class TilingData:
    def __init__(self, feature_width: int, feature_height: int, tile_size: int) -> None:
        self.feature_width = feature_width
        self.feature_height = feature_height
        self.tile_size = tile_size
    

def non_linear_func(val: float, factor: float = 0.75, start: float=0.1, end: float=0.9)->float:
    """A non linear function from the paper

    Args:
        val (float): the input value (tile value) value in [0,1]
        factor (float, optional): beta zero or factor, this describe the grafik of the value. Defaults to 0.25.
        start (float, optional): beta one, this is start value. Defaults to 0.1.
        end (float, optional): beta two, this is the end value. Defaults to 0.9.

    Returns:
        float: value in range of [start, end] or [end, start] if start > end
    """
    if (factor>=0.5):
        return start+(end-start)*(pow(val, 2*factor-1))
    else:
        return start+(end-start)*(pow(val, 1/(1-2*factor)))
    
def tiling_and_weight(tileData: TilingData, keypoints: list[cv2.KeyPoint], weight: float):
    feature_width = tileData.feature_width
    feature_height = tileData.feature_height
    tile_size = tileData.tile_size
    feature_map = [[0.0] * feature_width for _ in range(feature_height)]
    total = 0
    # Accumulate
    for point in keypoints:
        x, y = point.pt
        x//=tile_size;y//=tile_size
        feature_map[int(y)][int(x)]+=1
        total+=1
    # Normalize
    if (total>0):
        for i in range(feature_height):
            for j in range(feature_width):
                feature_map[i][j]*=(weight/total)
    return feature_map
    
def extract_feature(img: cv2.typing.MatLike, tile_size: int, weights: list[float])->list[list[float]] | None:
    height, width = img.shape[:2]
    if (width%tile_size!=0 or height%tile_size!=0):
        print(
            f"Error: Size of the image cannot be divided evenly with image width: {width} "
            f"and height: {height} on tile size of {tile_size}"
        )
        return 
    feature_width, feature_height = width//tile_size, height//tile_size
    tilingData = TilingData(
        feature_width=feature_width,
        feature_height=feature_height,
        tile_size=tile_size
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Get features
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kp_fast: list[cv2.KeyPoint] = fast.detect(gray, None)
    fm_fast = tiling_and_weight(tileData=tilingData, keypoints=kp_fast, weight=weights[0])
    # 2. Deteksi dengan STAR (membutuhkan opencv-contrib-python)
    star = cv2.xfeatures2d.StarDetector_create()
    kp_star: list[cv2.KeyPoint] = star.detect(gray, None)
    fm_star = tiling_and_weight(tileData=tilingData, keypoints=kp_star, weight=weights[1])
    # 3. Deteksi dengan ORB (sebagai pembanding)
    orb = cv2.ORB_create(nfeatures=500)
    kp_orb: list[cv2.KeyPoint] = orb.detect(gray, None)
    fm_orb = tiling_and_weight(tileData=tilingData, keypoints=kp_orb, weight=weights[2])
    # Merge
    comp_qual: list[list[float]] = [
        [
            non_linear_func(val=x+y+z)
            for x,y,z in zip(fm_fast[i], fm_star[i], fm_orb[i])
        ]
        for i in range(feature_height)
    ]
    
    return comp_qual

img = cv2.imread("sample4.png", cv2.IMREAD_GRAYSCALE)
WEIGHTS = [1/3,1/3,1/3] # Sum must equal 1 and not exceed 1
assert(sum(WEIGHTS)<=1)
assert(not (img is None))
print(extract_feature(img=img, tile_size=64, weights=WEIGHTS))