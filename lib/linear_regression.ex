defmodule PlayingWithNx.LinearRegression do
  import Nx.Defn

  defn calc_b(x_tensor, y_tensor) do
    top = Nx.sum(
      Nx.multiply(
        (x_tensor - Nx.mean(x_tensor)), 
        (y_tensor - Nx.mean(y_tensor))
      )
    )
    bottom = Nx.sum(
      Nx.power(
        (x_tensor - Nx.mean(x_tensor)), 
        2
      )
    )
    top/bottom
  end

  defn calc_a(x_tensor, y_tensor) do
      Nx.mean(y_tensor) - Nx.multiply(
        calc_b(x_tensor, y_tensor), 
        Nx.mean(x_tensor)
      )
  end

  def linear_regression(x_tensor, y_tensor) do
    %{:a => calc_b(x_tensor, y_tensor), :b => calc_a(x_tensor, y_tensor)}
  end
end