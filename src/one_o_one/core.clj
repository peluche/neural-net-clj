(ns one-o-one.core
  (:refer-clojure :exclude [* - + == /])
  (:require [clojure.contrib.generic.math-functions :as mathf]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [clojure.data.json :as json]))

(set-current-implementation :vectorz)

;; ==========
;; TODO-list
;; ==========
;; - add debug maccros to check that parameters are (matrix :vectorz)

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
  (matrix (array (for [i (range n)]
                   (for [j (range m)]
                     (rand))))))

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
  (spit filename (json/write-str (matrix :sequence net))))

(defn unserialize-neural-network
  "load a neural network from a file"
  [filename]
  (mapv matrix (json/read-str (slurp filename))))

(defn- add-biais-and-propagate
  "add the biais (set to 1) to the input and apply Theta
   then normalize it with the sigmoid function"
  [theta in]
  (let [biais    (matrix [[1]]) ;; i could shave some perf by making this global as it never changes
        in-biais (join biais in)
        out      (mmul theta in-biais)]
    ;; could use inplace modification (emap!) because:
    ;; if a tree fall and no one see it, it doesn't make noise !
    (emap activation-function out)))

(defn forward-propagation
  "run forward-propagation on a neural network with the given input
   the biais is automatically added (and thus should not be provided)"
  [[theta & net] in]
  (if (nil? theta)
    []
    ;; make sure "in" is a matrix ?
    (let [next-in (add-biais-and-propagate theta in)]
      (conj (forward-propagation net next-in) next-in))))

(defn cost-output
  "calculate the cost (error) on the output layer"
  [out expected]
  ;; make sure "expected" is a matrix ?
  (- out expected))


(defn- backward-propagation-rec
  "calculates the deltas for each layer"
  [[weights & net] [a & forward-proped] error]
  (if (nil? a)
    []
    (let [len              (dec (second (shape weights))) ;; to remove biais
          backward-weights (transpose (submatrix weights 1 [1 len]))
          t-theta-delta    (mmul backward-weights error)
          nodes-error   (* t-theta-delta
                           (emap derivative-activation-function a))]
      (conj (backward-propagation-rec net forward-proped nodes-error)
            nodes-error))))

(defn backward-propagation
  "run backward-propagation on the return of forward-propagation"
  [net [out & forward-propagated] expected]
  (let [out-error (cost-output out expected)]
    (conj (backward-propagation-rec (reverse net) forward-propagated out-error)
          out-error)))

(defn- biais-trans
  "add the biais (1) to a vector, transpose, and remove a dimension
   (because core.matrix doesn't broadcast well enough)"
  [v]
  ;; TODO: cast to matrix ?
  (let [biais (matrix [[1]])] ;; i could shave some perf by making this global as it never changes
    (join biais (first (transpose v)))))

(defn- el-wise-mul [A B]
  "TODO: replace me with a real element wise matrix/vector multiplication"
  ;; TODO: cast to matrix ?
  (transpose (* (transpose A) (first (transpose B)))))

(defn calc-delta-net
  "calculate the big delta weights of the neural-network after backprop"
  [[w & net] [d & deltas] [v & vals] learning-rate]
  (if (nil? w)
    '()
    (let [activation-levels (transpose (cons [1] v)) ;; TODO: cast to matrix?
          weights-change    (mmul d activation-levels)
          new-weights       (* learning-rate weights-change)]
      (conj (calc-delta-net net deltas vals learning-rate)
            new-weights))))

(defn helper-test-loop
  "run one exemple n times and output error"
  [in expected rate iterations initial-net]
  (let [in       (matrix in)
        expected (matrix expected)]
    (loop [net  initial-net
           iter iterations]
      (let [prop       (forward-propagation net in)
            deltas     (backward-propagation net prop expected)
            vals       (cons in (reverse prop))
            big-deltas (calc-delta-net net deltas vals rate)
            new-net    (map - net big-deltas)
            ;; _ (println "deltas 0:" (class (nth deltas 0)))
            ;; _ (println "deltas 1:" (class (nth deltas 1)))
            ;; _ (println "vals 0:" (class (nth vals 0)))
            ;; _ (println "vals 1:" (class (nth vals 1)))
            ;; _ (println "vals 2:" (class (nth vals 2)))
            ;; _ (println "big-deltas 0:" (class (nth big-deltas 0)))
            ;; _ (println "big-deltas 1:" (class (nth big-deltas 1)))
            ;; _ (println "new-net 0:" (class (nth new-net 0)))
            ;; _ (println "new-net 1:" (class (nth new-net 1)))
            ]
        ;; (pm (first prop))
        (if (> iter 0)
          (recur new-net (dec iter))
          prop)))))

(fn []

  (let [iter 100
        in  (array (repeat 400 [1]))
        out (array (repeat 400 [0.5]))
        net (make-neural-network [400 400 400])]
    (time (do
            (helper-test-loop in out 0.1 iter net)
            nil)))

  (let [iter 1
        in  (matrix (array (repeat 400 [1])))
        out (matrix (array (repeat 400 [0.5])))
        net (make-neural-network [400 400 400])
        prop (forward-propagation net in) ; [ goog type ]
        deltas (backward-propagation net prop out) ; first bad
        ]
    (time (println (class (nth deltas 1)))))

(defn aaa []
  (let [iter 1
        in  (matrix (array (repeat 400 [1])))
        out (matrix (array (repeat 400 [0.5])))
        net (make-neural-network [400 400 400])
        prop (forward-propagation net in) ; [ goog type ]
        deltas (backward-propagation net prop out) ; first bad
        ]
    (time (println (class (nth deltas 1)))))))


;; ==================
;; benchmark memory
;; ==================
;; (let [gc       (fn [] (dotimes [_ 4] (System/gc)))
;;       used-mem (fn []
;;                  (let [runtime (Runtime/getRuntime)]
;;                    (gc)
;;                    (- (.totalMemory runtime) (.freeMemory runtime))))
;;       before (used-mem)
;;       _      (make-neural-network [400 400 1000])
;;       after  (used-mem)]
;;   (println "mem: ~" (- after before 24) "bytes"))

;; (vector-of)
;; (keys (ns-publics 'clojure.core.matrix))
