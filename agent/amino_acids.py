# 性质值按 AAindex 残基顺序排列
AAINDEX_ORDER = "ARNDCQEGHILKMFPSTWYV"

SCALES = {
    "hydropathy": {
        "accession": "KYTJ820101",
        "citation": "Kyte and Doolittle, 1982; PMID:7108955",
        "description": "Kyte-Doolittle 疏水性指数，数值越大越疏水",
        "values": [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
                   3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2],
    },
    "polarity": {
        "accession": "GRAR740102",
        "citation": "Grantham, 1974; PMID:4843792",
        "description": "Grantham 极性尺度，数值越大表示此尺度上的极性越强",
        "values": [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2,
                   4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9],
    },
    "volume": {
        "accession": "GRAR740103",
        "citation": "Grantham, 1974; PMID:4843792",
        "description": "Grantham 体积尺度，用于比较残基大小，不是 GB1 中测得的体积",
        "values": [31, 124, 56, 54, 55, 85, 83, 3, 96, 111,
                   111, 119, 105, 132, 32.5, 32, 61, 170, 136, 84],
    },
    "charge": {
        "accession": "KLEP840101",
        "citation": "Klein, Kanehisa and DeLisi, 1984; PMID:6547351",
        "description": "简化净电荷编码；H 编码为 0，不代表任意 pH 下均不带电",
        "values": [0, 1, 0, -1, 0, 0, -1, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}

PROPERTIES = {
    aa: {name: scale["values"][index] for name, scale in SCALES.items()}
    for index, aa in enumerate(AAINDEX_ORDER)
}

SOURCES = {
    name: {
        "accession": scale["accession"],
        "citation": scale["citation"],
        "description": scale["description"],
        "url": f"https://www.genome.jp/dbget-bin/www_bget?aaindex:{scale['accession']}",
    }
    for name, scale in SCALES.items()
}
