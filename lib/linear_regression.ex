defmodule PlayingWithNx.LinearRegression do
  import Nx.Defn

  defn calc_beta(x_tensor, y_tensor) do
    numerator = Nx.sum(
      Nx.multiply(
        (x_tensor - Nx.mean(x_tensor)), 
        (y_tensor - Nx.mean(y_tensor))
      )
    )
    denominator = Nx.sum(
      Nx.power(
        (x_tensor - Nx.mean(x_tensor)), 
        2
      )
    )
    numerator/denominator
  end

  defn calc_alpha(x_tensor, y_tensor) do
      Nx.mean(y_tensor) - Nx.multiply(
        calc_beta(x_tensor, y_tensor), 
        Nx.mean(x_tensor)
      )
  end

  def linear_regression(x_tensor, y_tensor) do
    %{:beta => calc_beta(x_tensor, y_tensor), :alpha => calc_alpha(x_tensor, y_tensor)}
  end
end