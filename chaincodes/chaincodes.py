# internal imports
import warnings
from functools import reduce
from datetime import datetime
from typing import Union, List, Any

# external imports
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import sunpy
import sunpy.net
from sunpy.net.helio import Chaincode
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import numpy as np
import dfp
import astropy
import networkx
import pandas as pd


def to_chaincode(
    x: Union[int, float], y: Union[int, float], cc: str, cdelt1: float, cdelt2: float
) -> Chaincode:
    """Create a sunpy `Chaincode` instance from the chaincode representation.

    Create an instance of a sunpy `Chaincode` from the centre x, y
    pixel, the cc chaincode, and the scale in the horizontal and
    vertical dimensions.

    Parameters
    ----------
    x : Union[int, float]
        The starting x pixel coordinate for the chaincode.
    y : Union[int, float]
        The starting y pixel coordinate for the chaincode.
    cc : str
        The chaincode string representing the object.
    cdelt1 : float
        Plate scale in the x dimension.
    cdelt2 : float
        Plate scale in the y dimension.

    Examples
    --------
    >>> from chaincodes import to_chaincode
    >>> to_chaincode(1, 5, '23333525567', 1., 1.)
    Chaincode([1, 5])

    """
    return Chaincode([x, y], cc, xdelta=cdelt1, ydelta=cdelt2)


def dataframe_to_chaincodes(
    feature_df: pd.DataFrame,
) -> tuple[Chaincode]:
    """Create a list of `Chaincodes` from a pandas dataframe.

    Transform a pandas dataframe that describes many objects with
    their chaincodes, into a list of sunpy Chaincode objects.

    Parameters
    ----------
    feature_df : pd.DataFrame): tuple[sunpy.net.helio.Chaincode
        The pandas dataframe that has the following columns: cc_x_pix,
        cc_y_pix, cc

    Examples
    --------
    FIXME: Add docs.

    """
    x = feature_df.cc_x_pix.tolist()
    y = feature_df.cc_y_pix.tolist()
    c = feature_df.cc.tolist()
    cdelt1 = [1] * len(feature_df)
    cdelt2 = [1] * len(feature_df)
    return dfp.tmap(lambda r: to_chaincode(*r), zip(x, y, c, cdelt1, cdelt2))


def chaincode_to_skycoord(cc: Chaincode, smap: sunpy.map.GenericMap) -> SkyCoord:
    """Convert a `Chaincode` into a `SkyCoord`.

    Convert a `Chaincode` into `SkyCoord` given a particular map.

    Parameters
    ----------
    cc : Chaincode
        The chaincode to convert.
    smap : sunpy.map.GenericMap
        The sunpy map for which the SkyCoord will be projected onto
        the WCS of.

    Examples
    --------
    FIXME: Add docs.

    """
    x, y = cc.coordinates
    return SkyCoord.from_pixel(x, y, smap.wcs)


def to_skycoord(
    x: Union[int, float],
    y: Union[int, float],
    cc: str,
    cdelt1: float,
    cdelt2: float,
    obs_time: datetime,
) -> SkyCoord:
    return chaincode_to_skycoord(to_chaincode(x, y, cc, cdelt1, cdelt2), obs_time)


def dataframe_to_skycoords(
    feature_df: pd.DataFrame, smap: sunpy.map.GenericMap
) -> tuple[SkyCoord]:
    x = feature_df.cc_x_pix.tolist()
    y = feature_df.cc_y_pix.tolist()
    c = feature_df.cc.tolist()
    dates = feature_df.date_obs.tolist()
    smaps = []
    for d in dates:
        smap_header = smap.meta.copy()
        smap_header["date_obs"] = d
        smap_header["date-obs"] = d
        smap_copy = sunpy.map.Map(smap.data, smap_header)
        smaps.append(smap_copy)
    cdelt1 = [1] * len(feature_df)
    cdelt2 = [1] * len(feature_df)
    return dfp.tmap(
        lambda r: rotate_skycoord_to_map(chaincode_to_skycoord(r[0], r[1]), smap),
        zip(
            dfp.tmap(lambda r: to_chaincode(*r), zip(x, y, c, cdelt1, cdelt2)),
            smaps,
        ),
    )


def rotate_skycoord_to_map(
    skycoord: Union[SkyCoord, List[SkyCoord]], smap: sunpy.map.Map
) -> SkyCoord:
    if isinstance(skycoord, (tuple, list)):
        if len(skycoord) == 0:
            return []
        return [
            rotate_skycoord_to_map(dfp.first(skycoord), smap)
        ] + rotate_skycoord_to_map(dfp.rest(skycoord), smap)
    return rotate_skycoord_to_time(skycoord, timestamp=smap.date)


def rotate_skycoord_to_time(
    skycoord: SkyCoord, timestamp: astropy.time.Time
) -> SkyCoord:
    warnings.resetwarnings()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = solar_rotate_coordinate(skycoord, time=timestamp)
    return sk


def project_chaincode_to_world(cc: Chaincode, smap: sunpy.map.GenericMap):
    wcs = smap.wcs
    x, y = cc.coordinates
    coords = [(x[i], y[i]) for i in range(x.shape[0])]
    return wcs.wcs_pix2world(coords, 1)


def skycoord_to_pixel(skycoord, smap: sunpy.map.Map) -> tuple[np.ndarray, np.ndarray]:
    y, x = skycoord.to_pixel(smap.wcs)
    y = np.array(y)
    x = np.array(x)
    y = y[~np.isnan(y)].astype(np.int64)
    x = x[~np.isnan(x)].astype(np.int64)
    return x, y


def complete_outline(x, y) -> tuple[np.ndarray, np.ndarray]:
    def interpolate(x1, y1, x2, y2):
        # iterative interpolation between two integer coordinates:
        # produces every integer between these two points...could be
        # better but works.

        dist = int(np.floor(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)))
        if dist > 5000:
            dist = 1  # TODO: strange overflow for something that should be one:
            # print(x1, y1, x2, y2, dist)
            # 342 179 343 178 18446744073709551616
        if dist:
            step_x = int(round((x2 - x1) / dist))
            step_y = int(round((y2 - y1) / dist))
            _x = x1
            _y = y1
            for _ in range(dist - 1):
                _x += step_x
                _y += step_y
                yield _x, _y

    out_x, out_y = [x[0]], [y[0]]
    for idx in range(1, x.shape[0]):
        for xi, yi in interpolate(x[idx - 1], y[idx - 1], x[idx], y[idx]):
            out_x.append(xi)
            out_y.append(yi)
        out_x.append(x[idx])
        out_y.append(y[idx])

    out_x = np.array(out_x).astype(np.int64)
    out_y = np.array(out_y).astype(np.int64)
    return out_x, out_y


def infill_outline(outline: np.ndarray) -> np.ndarray:
    return scipy.ndimage.binary_fill_holes(outline).astype(int)


def diff(a, which="horizontal") -> np.ndarray:
    if which == "both":
        return ((diff(a, which="horizontal") + diff(a, which="vertical")) > 0.0).astype(
            np.float64
        )

    out = np.zeros_like(a)
    for row_idx in range(1, a.shape[0]):
        for col_idx in range(1, a.shape[1]):
            if which == "horizontal":
                if a[row_idx, col_idx - 1] != a[row_idx, col_idx]:
                    if a[row_idx, col_idx - 1] == 0:
                        out[row_idx, col_idx] = 1.0
                    else:
                        out[row_idx, col_idx - 1] = 1.0
            elif which == "vertical":
                if a[row_idx - 1, col_idx] != a[row_idx, col_idx]:
                    if a[row_idx - 1, col_idx] == 0:
                        out[row_idx, col_idx] = 1.0
                    else:
                        out[row_idx - 1, col_idx] = 1.0
    return out


def chaincode_to_mask(
    coord: sunpy.net.helio.Chaincode, smap: sunpy.map.GenericMap
) -> np.ndarray:
    x, y = coord.coordinates
    mask = np.zeros_like(smap.data)
    mask[y.astype(int), x.astype(int)] = 1.0
    mask = infill_outline(mask)
    return mask


def skycoord_to_mask(
    skycoord: astropy.coordinates.SkyCoord, smap: sunpy.map.GenericMap
) -> np.ndarray:
    x, y = skycoord_to_pixel(skycoord, smap)
    output = np.zeros(smap.data.shape)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return output
    x, y = complete_outline(x, y)
    x[x >= output.shape[0]] = output.shape[0] - 1
    y[y >= output.shape[1]] = output.shape[1] - 1
    output[x, y] = 1.0
    output = infill_outline(output)
    return output


def skycoords_to_mask(
    skycoords: Union[tuple[SkyCoord], list[SkyCoord]], smap: sunpy.map.GenericMap
) -> np.ndarray:
    return reduce(
        lambda t, x: skycoord_to_mask(x, smap) + t,
        dfp.rest(skycoords),
        skycoord_to_mask(dfp.first(skycoords), smap),
    )


def dataframe_to_mask(
    feature_df: pd.DataFrame, smap: sunpy.map.GenericMap
) -> np.ndarray:
    pipeline = dfp.compose(
        lambda df: dataframe_to_skycoords(df, smap),
        lambda sk: skycoords_to_mask(sk, smap),
    )
    return pipeline(feature_df)


def build_chain(outline: np.ndarray) -> dict[str, Any]:
    # generate a freeman 8 component chain code
    def get_neighbours(x, y):
        connected_neighbours: list[tuple[int, int]] = []
        for xi in range(max(x - 1, 0), min(x + 2, outline.shape[0])):
            for yi in range(max(y - 1, 0), min(y + 2, outline.shape[1])):
                if (xi != x or yi != y) and outline[xi, yi] == 1.0:
                    connected_neighbours.append((xi, yi))
        return connected_neighbours

    graph = networkx.Graph()
    indices = np.where(outline == 1.0)
    for xy in zip(indices[0], indices[1]):
        graph.add_node(xy)
        for n in get_neighbours(xy[0], xy[1]):
            graph.add_edge(xy, n)

    code = []
    chain = reduce(
        lambda t, x: x if len(x) > len(t) else t,
        networkx.chain_decomposition(graph),
        next(iter(networkx.chain_decomposition(graph))),
    )
    for link in chain:
        last_x, last_y = link[0]
        xi, yi = link[1]
        if xi - last_x == 1 and yi - last_y == 0:
            code.append("6")
        if xi - last_x == 1 and yi - last_y == 1:
            code.append("5")
        if xi - last_x == 0 and yi - last_y == 1:
            code.append("4")
        if xi - last_x == -1 and yi - last_y == 1:
            code.append("3")
        if xi - last_x == -1 and yi - last_y == 0:
            code.append("2")
        if xi - last_x == -1 and yi - last_y == -1:
            code.append("1")
        if xi - last_x == 0 and yi - last_y == -1:
            code.append("0")
        if xi - last_x == 1 and yi - last_y == -1:
            code.append("7")

    return {
        "cc_x_pix": chain[0][1][0],
        "cc_y_pix": chain[0][1][1],
        "cc": "".join(code),
        "cc_length": len(code),
    }


def pixel_to_arcsec(
    x: Union[int, float], y: Union[int, float], smap: sunpy.map.GenericMap
) -> tuple[float, float]:
    s = smap.pixel_to_world(x * u.pixel, y * u.pixel)
    return s.Tx.value, s.Ty.value


def mask_to_chaincode(mask: np.ndarray, smap: sunpy.map.Map) -> pd.DataFrame:
    chaincodes: list[dict[str, Any]] = []
    labelled_array, n_features = scipy.ndimage.label(mask)
    for feature_idx in range(1, n_features + 1):
        _tmp = labelled_array == feature_idx
        outline = diff((_tmp == 1).astype(np.int8), "both")
        code = build_chain(outline)
        chaincodes.append(code)
    df = pd.DataFrame(chaincodes)
    df["cdelt1"] = smap.meta["cdelt1"]
    df["cdelt2"] = smap.meta["cdelt2"]
    df["cc_x_arcsec"] = 0.0
    df["cc_y_arcsec"] = 0.0
    for row_idx in range(df.shape[0]):
        x, y = df.loc[row_idx, "cc_x_pix"], df.loc[row_idx, "cc_y_pix"]
        x, y = pixel_to_arcsec(x, y, smap)
        df.loc[row_idx, "cc_x_arcsec"] = x
        df.loc[row_idx, "cc_y_arcsec"] = y
    df["date_obs"] = smap.date.strftime("%Y-%m-%dT%H:%M:%S")  # isoformat
    return df
