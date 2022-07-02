defmodule PlayingWithNx.InverseMatrix do
  import Nx.LinAlg
  import Nx.Defn

  def generate_square_matrix(n) do
    for _i <- 1..n do
      for _j <- 1..n do
        :rand.uniform()
      end
    end
    |> Nx.tensor()
  end

  def time_inverse(n) do
    a = generate_square_matrix(n)

    {t_micro, :ok} =
      :timer.tc(fn ->
        Nx.LinAlg.invert(a)
        :ok
      end)

    t_micro / 1_000_000
  end
end
