(ns one-o-one.core-test
  (:require [clojure.test :refer :all]
            [one-o-one.core :refer :all]
            [clojure.contrib.generic.math-functions :refer [approx=]]
            [clojure.core.matrix :refer [shape mget matrix]]))

(deftest activation-function-test
  (testing "activation-function tend to ]0-1["
    (is (approx= (activation-function 10) 1 1e-4))
    (is (approx= (activation-function -10) 0 1e-4))))

(deftest gradient-checking-test
  (testing "testing if our aproximation of derivative work"
    (let [activ (fn [x] (* x x))]
      (is (approx= (gradient-checking 1 1e-2 activ) 2 1e-8)))))

(deftest derivative-activation-function-test
  (testing "compare gradient-checking with derivative at ~1e-8"
    (let [x 2
          y (activation-function x)]
      (is (approx= (derivative-activation-function y)
                   (gradient-checking x 1e-4 activation-function)
                   1e-8)))
    (let [x 8
          y (activation-function x)]
      (is (approx= (derivative-activation-function y)
                   (gradient-checking x 1e-4 activation-function)
                   1e-8)))
    (let [x -5
          y (activation-function x)]
      (is (approx= (derivative-activation-function y)
                   (gradient-checking x 1e-4 activation-function)
                   1e-8)))
    (let [x 0
          y (activation-function x)]
      (is (approx= (derivative-activation-function y)
                   (gradient-checking x 1e-4 activation-function)
                   1e-8)))))

(deftest make-random-matrix-test
  (testing "making a matrix of dimensions m x n"
    (let [m 4
          n 5
          my-matrix (make-random-matrix m n)
          el1 (mget my-matrix 0 0)
          el2 (mget my-matrix 0 1)]
      (is (= [4 5] (shape my-matrix)))
      (is (not= el1 el2)))))

(deftest make-neural-network-test
  (testing "making a neural network (generate sevral matrices)"
    (let [in     2
          hidden 3
          out    4
          net    (make-neural-network [in hidden out])]
      (is (= 2 (count net)))
      (is (= [hidden (inc in)] (shape (nth net 0))))
      (is (= [out (inc hidden)] (shape (nth net 1)))))))

(deftest serialize-unserialize-test
  (testing "serialize then unserialize a neural network to disk"
    (let [net      (make-neural-network [2 3 4])
          filename "/tmp/serial-neural-net.test"
          _        (serialize-neural-network filename net)
          new-net  (unserialize-neural-network filename)]
      (is (= net new-net)))))

;;; forward-propagation-only
;; (deftest forward-propagation-only-test
;;   (testing "forward propagation only"
;;     (with-redefs [activation-function (fn [x] (* 2 x))]
;;       (let [in         [[2]
;;                         [3]]
;;             net-layer1 [[4  5  6]
;;                         [7  8  9]
;;                         [10 11 12]]
;;             net-layer2 [[13 14 15 16]
;;                         [17 18 19 20]]
;;             net        [net-layer1 net-layer2]
;;             wanted     [[9170]
;;                         [11578]]
;;             res        (forward-propagation-only net in)]
;;         (is (= res wanted))))))

(deftest forward-propagation-test
  (testing "forward propagation"
    (with-redefs [activation-function (fn [x] (* 2 x))]
      (let [in         [[2]
                        [3]]
            net-layer1 [[4  5  6]
                        [7  8  9]
                        [10 11 12]]
            net-layer2 [[13 14 15 16]
                        [17 18 19 20]]
            net        [net-layer1 net-layer2]
            wanted     [[9170.0]
                        [11578.0]]
            ret        (forward-propagation net in)
            res        (first ret)]
        (is (= res wanted))
        (is (= 2 (count ret)))))))

(deftest cost-output-test
  (testing "cost-output"
    (let [out      [[0.1] [0.2] [0.75]]
          expected [[0] [0] [1]]]
      (is (= [[0.1] [0.2] [-0.25]] (cost-output out expected))))))

(deftest backward-propagation-test
  (testing "backward propagation"
    (with-redefs [derivative-activation-function (fn [y] (* -2 y))]
      (let [in          [[2]
                         [3]]
            net-layer1  [[40 50 60]
                         [70 80 90]
                         [10 11 12]]
            net-layer2  [[10 1 2 3]
                         [20 4 5 6]]
            net         [net-layer1 net-layer2]
            forward-res [[[1] [8]]
                         [[7] [8] [9]]]
            expected    [[0] [10]]
            wanted      [[[98] [128] [162]]
                         [[1] [-2]]]
            ret         (backward-propagation net forward-res expected)]
        (is (= wanted ret))))))

;; (deftest calc-delta-net-test
;;   (testing "update the net weights"
;;     (let [learning-rate  2 ;; i want to keep results as integers
;;           in             [[2]
;;                           [3]]
;;           net-layer1     [[40 50 60]
;;                           [70 80 90]
;;                           [10 11 12]]
;;           net-layer2     [[10 1 2 3]
;;                           [20 4 5 6]]
;;           net            [net-layer1 net-layer2]
;;           deltas         [[[21] [31] [41]]
;;                           [[4] [7]]]
;;           forward-proped [[[9170] [11578]] [[64] [100] [136]]]
;;           vals           (cons in (reverse forward-proped))
;;           ret           (calc-delta-net net deltas vals learning-rate)
;;           expected      [[[1680    4200    7560]
;;                           [4340    9920   16740]
;;                           [ 820    1804    2952]]

;;                          [[80     512    1600    3264]
;;                           [280    3584    7000   11424]]]]
;;       (is (= ret expected)))))

(deftest loop-it-baby
  (testing "non-regression test"
    (let [net  [[[0.44 0.18 0.32]
                 [0.40 0.24 0.29]
                 [0.64 0.45 0.62]]
               
                [[0.72 0.71 0.15 0.03]
                 [0.28 0.94 0.11 0.07]
                 [0.64 0.53 0.06 0.82]
                 [0.92 0.03 0.40 0.59]]]
          in   [[1] [0]]
          out  [[0] [1] [1] [0]]
          rate 0.3
          iter 15
          ret  (helper-test-loop in out 0.3 iter net)]
      (is (= ret
             [[[0.09463351132639372] [0.9371167956242776] [0.9469569426446957] [0.0969249559309502]]
              [[0.7533369989150758] [0.7192067469930032] [0.8125626854366405]]])))))

;; ==========
;; slow test
;; ==========
;; (deftest learn-test
;;   (testing "learning the XOR function"
;;     (let [gen-xor-dataset  (fn []
;;                              (let [in1 (Math/round (rand))
;;                                    in2 (Math/round (rand))
;;                                    out (bit-xor in1 in2)]
;;                                [[in1 in2] [out]]))
;;           approx-col=      (fn [a b eps]
;;                              (reduce #(and %1 %2)
;;                                      (map (fn [m n] (approx= (first m) n eps)) a b)))
;;           dataset          (repeatedly 10000 gen-xor-dataset)
;;           net              (make-neural-network [2 3 1])
;;           trained-net      (train net dataset 0.2)]
;;       (is (approx-col= (run-input trained-net [[0] [0]]) [0] 0.1))
;;       (is (approx-col= (run-input trained-net [[0] [1]]) [1] 0.1))
;;       (is (approx-col= (run-input trained-net [[1] [0]]) [1] 0.1))
;;       (is (approx-col= (run-input trained-net [[1] [1]]) [0] 0.1)))))

