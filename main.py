import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Tuple

# 乱数のシードを固定（再現性確保）
np.random.seed(42)

# 訓練データ・テストデータの件数
n_train: int = 800_000
n_test: int = 200_000

# データセット生成時の乱数の上限・下限
default_high: float = 1.0
default_low: float = -1.0

# モデル内部の式の精度（この値より小さい数値はゼロとみなす）
eps: float = 1e-3


def true_func(X: np.ndarray) -> np.ndarray:
    """
    真の関数を計算して返す

    Args:
        X (np.ndarray): 入力データ (shape: [n, 2]) 列順は [a, m]

    Returns:
        np.ndarray: 出力データ (shape: [n, 1])
    """
    return (X[:, 0] * X[:, 1]).reshape(-1, 1)


def make_dataset(
    n: int,
    low: float = default_low,
    high: float = default_high,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    一様分布に従う乱数からデータセットを生成する。

    Args:
        n (int): サンプル数
        low (float): 最小値（inclusive）
        high (float): 最大値（exclusive）

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: 形状 (n, 2) の説明変数 [a, m]
            - y: true_func(X) によって生成される目的変数
    """
    X: np.ndarray = np.random.uniform(low, high, size=(n, 2))
    y: np.ndarray = true_func(X)
    return X, y


def main() -> None:
    """
    ニューラルネットワークモデルを学習し、
    テストデータで評価し、
    学習結果およびモデルから推定された真の関数の式を出力する。
    """
    # データ生成
    x_train, y_train = make_dataset(n_train)
    x_test, y_test = make_dataset(n_test)

    #  モデル定義: MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(50, 50),  # 隠れ層２層×50ユニット
        activation="relu",  # 活性化関数
        solver="adam",  # 最適化アルゴリズム
        batch_size=1024,  # ミニバッチサイズ
        learning_rate_init=1e-3,  # 初期学習率
        max_iter=50,  # エポック数
        random_state=42,
        verbose=True,
    )

    # モデル学習
    model.fit(x_train, y_train.ravel())

    # 予測・評価
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"テストデータに対する平均絶対誤差 (MAE): {mae:.6f}")
    if mae <= 0.04:
        print("精度: OK")
    else:
        print("精度: 不十分です")

    """
    モデル出力から真の関数を近似的に再構築
    """
    # テストデータのサブセットを使って予測曲面を線形回帰でフィッティング
    n_sub: int = 200_000
    idx = np.random.choice(len(x_test), size=n_sub, replace=False)
    X_sub = x_test[idx]
    y_sub_pred = model.predict(X_sub)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_sub)
    lr = LinearRegression()
    lr.fit(X_poly, y_sub_pred)

    # 特徴量名と係数取得
    feature_names = poly.get_feature_names_out(["a", "m"])
    coefs = lr.coef_
    intercept = lr.intercept_

    # フルモデル式
    terms = [(name.replace(" ", "*"), w) for name, w in zip(feature_names, coefs)]
    full = " + ".join(f"{w:.6f}*{n}" for n, w in terms)
    full = f"{full} + {intercept:.6f}"

    # 省略版: 絶対値 eps 未満の項は省略
    filtered = " + ".join(f"{w:.6f}*{n}" for n, w in terms if abs(w) >= eps)
    if abs(intercept) >= eps:
        filtered = f"{filtered} + {intercept:.6f}" if filtered else f"{intercept:.6f}"

    print("\n【フルモデル式（近似）】")
    print(f"  ŷ = {full}")
    print("\n【省略版モデル式（近似）】")
    print(f"  ŷ = {filtered}")


if __name__ == "__main__":
    main()
