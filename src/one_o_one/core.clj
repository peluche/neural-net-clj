(ns one-o-one.core
  (:require [clojure.contrib.generic.math-functions :as mathf]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [clojure.data.json :as json]))

(defn activation-function
  "the sigmoid activation function"
  [x]
  (/ 1 (+ 1 (mathf/exp (- x)))))

(defn derivative-activation-function
  "derivative of the activation function for back-propagation
   note: 'y' is a vector, (corresponding to 'a^(super-script)'
         '1' is a vector of 1s [[1] [1] ... [1]] (thanks to broadcast)
         y .* (1 - y) is an element-wise multiplication
         this formula is equivalent to g'(z^(supper-script))"
  [y]
  (* y (- 1 y)))

(defn gradient-checking
  "calculate an approximate of the derivative
   (for testing purpose, don't use for real learning)"
  [x epsilon activ]
  (/ (- (activ (+ x epsilon)) (activ (- x epsilon))) (* 2 epsilon)))

(defn make-random-matrix
  "return a (n x m) matrix filled with random values"
  [n m]
  (array (for [i (range n)]
           (for [j (range m)]
             ;; TODO: check if a ZERO weight would break the algo
             (+ 1e-5 (rand))))))

(defn make-neural-network
  "return a representation of a neural network
  ex: [4 5 3]
  4 features (inputs)
  1 hidden layer with 5 nodes
  3 outputs"
  [dimensions]
  (map (fn [[n m]] (make-random-matrix m (inc n))) (partition 2 1 dimensions)))

(defn serialize-neural-network
  "save a neural network to a file"
  [filename net]
  (spit filename (json/write-str net)))

(defn unserialize-neural-network
  "load a neural network from a file"
  [filename]
  (json/read-str (slurp filename)))

(defn- add-biais-and-propagate
  "add the biais (set to 1) to the input and apply Theta
   then normalize it with the sigmoid function"
  [theta in]
  (let [in-biais (array (cons [1] in))]
    ;; TODO: refactor with mapv
    (map (fn [[x]] [(activation-function x)]) (mmul theta in-biais))))

;; (defn forward-propagation-only
;;   "run forward-propagation on a neural network with the given input
;;    the biais is automatically added (and thus should not be provided)"
;;   [[theta & net] in]
;;   (if (nil? theta)
;;     in
;;     (recur net (add-biais-and-propagate theta in))))

(defn forward-propagation
  "run forward-propagation on a neural network with the given input
   the biais is automatically added (and thus should not be provided)"
  [[theta & net] in]
  (if (nil? theta)
    []
    (let [next-in (add-biais-and-propagate theta in)]
      (conj (forward-propagation net next-in) next-in))))

(defn cost-output
  "calculate the cost (error) on the output layer"
  [out expected]
  (- out expected))

(defn- backward-propagation-rec
  "calculates the deltas for each layer"
  [[weights & net] [a & forward-proped] deltas]
  (if (nil? a)
    '()
    (let [t-theta-delta (rest (mmul (transpose weights) deltas))
          nodes-error   (* t-theta-delta
                           ;; TODO: refactor with mapv
                           (map (fn [[y]] [(derivative-activation-function y)]) a))]
      (conj (backward-propagation-rec net forward-proped nodes-error)
            nodes-error))))

(defn backward-propagation
  "run backward-propagation on the return of forward-propagation"
  [net [out & forward-propagated] expected]
  (let [out-error (cost-output out expected)]
    (conj (backward-propagation-rec (reverse net) forward-propagated out-error)
          out-error)))


;; (pm (make-random-matrix 4 5))
;; (def net (make-neural-network [4 3 2]))
;; (serialize-neural-network "/tmp/serial-neural-net.test" (inc 1e-12))
;; (unserialize-neural-network "/tmp/serial-neural-net.test")
;; ((fn [[theta & net]] net) nil)



