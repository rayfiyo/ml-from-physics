import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

# 乱数のシードを固定（再現性確保）
np.random.seed(42)


# 訓練データ・テストデータの件数
n_train: int = 8_000_000
n_test: int = 2_000_000

# データセット生成時の乱数の下限・上限
default_low: float = -100.0
default_high: float = +100.0


def true_func(x: np.ndarray) -> np.ndarray:
    """
    真の関数 y = 2x を計算して返す

    Args:
        x (np.ndarray): 入力データ (shape: [n, 1])
    Returns:
        np.ndarray: 出力データ (shape: [n, 1])
    """
    return 2.0 * x


def make_dataset(
    n: int, low: float = default_low, high: float = default_high
) -> Tuple[np.ndarray, np.ndarray]:
    """
    一様分布に従う乱数からデータセットを生成する。

    Args:
        n (int): サンプル数
        low (float): 最小値（inclusive）
        high (float): 最大値（exclusive）
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - x: 形状 (n, 1) の説明変数
            - y: true_func(x) によって生成される目的変数
    """
    # x は (n,1) の列ベクトルとしてサンプリング
    x: np.ndarray = np.random.uniform(low, high, size=(n, 1))
    y: np.ndarray = true_func(x)
    return x, y


def main() -> None:
    """
    線形回帰モデルを学習し、テストデータで評価を行い結果を出力する。

    - 学習データ生成 (make_dataset)
    - LinearRegression でフィッティング
    - 重み・バイアスの取得
    - テストデータで予測し MAE を計算
    - 平均絶対誤差が 0.04 以下かを判定
    """
    # 学習データ生成
    x_train, y_train = make_dataset(n_train)

    # モデル定義 & 学習
    model: LinearRegression = LinearRegression()
    model.fit(x_train, y_train)

    # 学習結果取得
    # coef_ は ndarray なので[0][0]でスカラにアクセス
    weight: float = float(model.coef_[0][0])
    bias: float = float(model.intercept_[0])

    # テストデータ生成 & 予測・評価
    x_test, y_test = make_dataset(n_test)
    y_pred: np.ndarray = model.predict(x_test)
    mae: float = float(np.mean(np.abs(y_test - y_pred)))

    # 結果出力
    print(f"学習で得られた重み: {weight:.4f}")
    print(f"学習で得られたバイアス: {bias:.4f}")
    print(f"テストデータに対する平均絶対誤差: {mae:.4f}")
    if mae <= 0.04:
        print("精度: OK")
    else:
        print("精度: 不十分です")


if __name__ == "__main__":
    main()
