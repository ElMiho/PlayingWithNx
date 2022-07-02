defmodule PlayingWithNx.ExponentialRegression do
  import Nx.Defn

  defn exp_to_linear(y_tensor) do
    Nx.log(y_tensor)
  end

  def exponential_regression(x_tensor, y_tensor) do
    %{:alpha => alpha, :beta => beta} =
      PlayingWithNx.LinearRegression.linear_regression(
        x_tensor,
        exp_to_linear(y_tensor)
      )

    %{:beta => Nx.exp(beta), :alpha => Nx.exp(alpha)}
  end
end
