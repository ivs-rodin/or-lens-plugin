\ Two plants ship to two warehouses that deliver to three customers.
\ Customers demand 75 units in total, but the plants can make only 70.
Minimize
 cost: 4 ship(P1_W1) + 6 ship(P1_W2) + 5 ship(P2_W1) + 3 ship(P2_W2)
  + 2 deliver(W1_C1) + 3 deliver(W1_C2) + 4 deliver(W2_C2) + 2 deliver(W2_C3)
Subject To
 capacity(P1): ship(P1_W1) + ship(P1_W2) <= 40
 capacity(P2): ship(P2_W1) + ship(P2_W2) <= 30
 balance(W1): ship(P1_W1) + ship(P2_W1) - deliver(W1_C1) - deliver(W1_C2) = 0
 balance(W2): ship(P1_W2) + ship(P2_W2) - deliver(W2_C2) - deliver(W2_C3) = 0
 demand(C1): deliver(W1_C1) >= 25
 demand(C2): deliver(W1_C2) + deliver(W2_C2) >= 30
 demand(C3): deliver(W2_C3) >= 20
 storage(W1): ship(P1_W1) + ship(P2_W1) <= 60
 storage(W2): ship(P1_W2) + ship(P2_W2) <= 60
 budget: 4 ship(P1_W1) + 6 ship(P1_W2) + 5 ship(P2_W1) + 3 ship(P2_W2) <= 2000
End
