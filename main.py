"""
機械学習を用いて物理法則を近似し、そのモデルの重みとバイアスを見ることで新しい近似公式を見るける実験
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# 乱数のシードを固定（再現性確保）
np.random.seed(42)

# 訓練データ・テストデータの件数
n_train: int = 16_000_000
n_test: int = 4_000_000

# 入力の未知数の数。（説明変数の形状の例: (n_train, n_features)）
# objective_function の式と合わせること！
n_features: int = 1

# データセット生成時の乱数の上限・下限
high: float = 1000.0
low: float = -1000.0


def objective_function(X: np.ndarray) -> np.ndarray:
    """
    目的関数（真の関数）を計算して返す
    n_features と合わせること！
    """
    # 2*x: n_features=1, -100.0 ~ 100.0, 隠れ層(2), エポック100 で収束
    return (2 * X[:, 0]).reshape(-1, 1)

    # 位置エネルギー U_g = mgh: n_features=2, 隠れ層(1, 1), エポック6000 で未収束
    # return (X[:, 0] * 9.8 * X[:, 1]).reshape(-1, 1)

    # 熱力学第一法則 Q = ΔU + W: n_features=2, 隠れ層(1, 1), エポック6000 で未収束
    # return (X[:, 0] + X[:, 1]).reshape(-1, 1)


def make_dataset(
    n: int,  # サンプル数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    一様分布に従う乱数からデータセットを生成する。
    """
    # 形状 (n, x_num) の説明変数
    X: np.ndarray = np.random.uniform(low, high, size=(n, n_features))

    # objective_function(X) によって生成される目的変数
    y: np.ndarray = objective_function(X)

    return X, y


def print_mlp_expression(
    model: MLPRegressor,
    feature_names: Tuple[str, ...],
):
    """
    model.coefs_, model.intercepts_ をたどり、
    ReLU を含むネットワークの出力式を組み立てて標準出力する。
    """
    coefs = model.coefs_
    intercepts = model.intercepts_
    n_layers = len(coefs)  # 隠れ層＋出力層の数

    # 層ごとに入力名リストを更新していく
    input_names = list(feature_names)

    for layer_idx, (W, b) in enumerate(zip(coefs, intercepts)):
        is_output_layer = layer_idx == n_layers - 1
        n_out = W.shape[1]

        # この層の出力を表す名前リスト
        output_names = []
        for j in range(n_out):
            # 線形結合の文字列を作成
            terms = []
            for i, xname in enumerate(input_names):
                terms.append(f"{W[i, j]:.3f}*{xname}")
            terms.append(f"{b[j]:.3f}")
            lin = " + ".join(terms) or "0"

            if not is_output_layer:
                # 隠れ層：ReLU
                neuron_name = f"h{layer_idx+1}_{j+1}"
                print(f"{neuron_name} = relu({lin})")
                output_names.append(neuron_name)
            else:
                # 出力層：恒等関数 → ŷ
                print(f"ŷ = {lin}")

        # 次の層の入力名リストを更新（出力層なら不要）
        if not is_output_layer:
            input_names = output_names


def main() -> None:
    """
    ニューラルネットワークモデルを学習し、
    テストデータで評価し、
    学習結果およびモデルから推定された真の関数の式を出力する。
    """
    # データ生成
    x_train, y_train = make_dataset(n_train)
    x_test, y_test = make_dataset(n_test)
    x_sample, y_sample = make_dataset(1)

    # 入力・出力を標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_s = scaler_x.fit_transform(x_train)
    y_train_s = scaler_y.fit_transform(y_train).ravel()
    x_test_s = scaler_x.transform(x_test)

    #  モデル定義と学習: MLPRegressor
    model = MLPRegressor(
        activation="relu",  # 活性化関数
        batch_size=1024,  # ミニバッチサイズ
        hidden_layer_sizes=(2),  # 隠れ層: ユニット数（各値）は 2 以上
        learning_rate="adaptive",
        learning_rate_init=1e-2,  # 初期学習率
        max_iter=16,  # エポック数
        random_state=42,  # ランダムのシード
        solver="adam",  # 最適化アルゴリズム
        tol=1e-3,  # 収束許容誤差を厳しく
        verbose=False,  # 学習の進行状況
    )
    model.fit(x_train_s, y_train_s)

    # 予測 + スケール変換 + MAE 算出
    y_pred_s = model.predict(x_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1))
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.6f}")

    # 演算の例（n=1）
    true_sample = objective_function(x_sample)[0][0]
    prediction_sample = model.predict(x_sample)[0]
    error = abs(true_sample - prediction_sample)
    print(
        f"入力例: {x_sample[0][0]:.6f}\n",
        f"  理論値: {true_sample:.6f}, 予測値: {prediction_sample:.6f}, 誤差: {error:.6f}",
    )

    # ネットワーク式の出力
    print("\n=== ネットワークが学習した式（ReLU込み） ===")
    print_mlp_expression(model, feature_names=("a"))


if __name__ == "__main__":
    main()
