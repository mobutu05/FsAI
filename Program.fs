// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open System

//args
let numIters = 1000
let numEps = 100
let tempThreshold = 15
let numMCTSSims = 25

type Board(n) =
    let directions = [(1,1);(1,0);(1,-1);(0,-1);(-1,-1);(-1,0);(-1,1);(0,1)]
    let n = n
    //Create the empty board array.
    //member this.pieces:int list list = [for i in 1..n do yield [for i in 1..n -> 0]]
    member this.is_win (board:int list list) color = 
        //Check whether the given player has collected a triplet in any direction;
        //@param color (1=white,-1=black)        
        let win = n
        let mutable count = 0
        // check y-strips
        for y in  0..(n-1) do
            count <- 0
            for x in 0..(n-1) do
                if board.[x].[y] = color then
                    count <- count + 1
            
            
        // check x-strips
        if count <> win then
            for x in 0..(n-1) do
                count <- 0
                for y in 0..(n-1) do
                    if board.[x].[y] = color then
                        count <- count + 1
        // check two diagonal strips
        if count <> win then
            count <- 0
            for d in 0..(n-1) do
                if board.[d].[d] = color then
                    count <- count + 1

        if count <> win then
            count <- 0
            for d in 0..(n-1) do
                if board.[d].[n - d - 1] = color then
                    count <- count + 1
        count = win

    member this.has_legal_moves (board:int list list) =
        List.fold (fun acc x -> List.fold (fun acc xx -> acc || (xx = 0) ) acc x ) false board   
        
type Game(n) =
    let n = 3
    member this.getBoardSize = 
        n, n
    member this.getActionSize = 
        n * n

    member this.getInitBoard =        
        [for i in 1..n do yield [for i in 1..n -> 0]]

    member this.getCanonicalForm board player =
        //return state if player==1, else return -state if player==-1
        List.map (fun x -> List.map (fun xx -> xx * player) x) board

    member this.stringRepresentation board =
        //8x8 numpy array (canonical board)
        List.fold (fun acc x -> List.fold (fun acc xx -> acc + (string xx) ) acc x ) "" board   
        
    member this.gameEnded board player =
        let b = Board(n)
        //return 0 if not ended, 1 if player 1 won, -1 if player 1 lost        
        if b.is_win board player then
            1.0
        elif b.is_win board -player then
            -1.0
        elif b.has_legal_moves board then
            0.0
        else//draw has a very little value
            1e-4

     //member this.get_legal_moves board color =
        ////Returns all the legal moves for the given color.
        ////(1 for white, -1 for black)
        ////@param color not used and came from previous version.
        
        //let moves = set()  //stores the legal moves.
        
        //// Get all the empty squares (color==0)
        //for y in range(self.n):
        //    for x in range(self.n):
        //        if self[x][y]==0:
        //            newmove = (x,y)
        //            moves.add(newmove)
        //return list(moves)
    member this.getValidMoves (board: int list list)  player = 
        // return a fixed size binary vector
        seq {for i in 0..n-1 do 
             for j in 0..n-1 do 
             if board.[i].[j] = 0 then yield 1.0 else yield 0.0} |> Seq.toList
        //let b = Board()
        //valids = [ for i in 0 .. self.getActionSize() -> 0.0]
        //legalMoves =  b.get_legal_moves(player)
        //if len(legalMoves)==0:
        //    valids[-1]=1
        //    return np.array(valids)
        //for x, y in legalMoves:
        //    valids[self.n*x+y]=1
        //return np.array(valids)



type NeuralNet(g:Game) =
    let board_x, board_y = g.getBoardSize
    let action_size = g.getActionSize
    member this.predict  =
        //board: np array with board
        
        //timing
        //start = time.time()

        //preparing input
       // board = board[np.newaxis, :, :]

        
        //pi, v = self.model.predict(board)
        //(List.take 10, 0)
         ([for i in 1.. 9 -> 1.0], 0.0)
        //print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        //return pi[0], v[0]
    
type MCTS(game, nnet) =
    let game:Game = game
    let nnet:NeuralNet = nnet
    let Qsa = [] //stores Q values for s,a (as defined in the paper)
    let Nsa = [] //stores #times edge s,a was visited
    let mutable Ns = Map.empty  //stores #times board s was visited
    let mutable Ps = Map.empty  //stores initial policy (returned by neural net)
    let mutable Es = Map.empty  //stores game.getGameEnded ended for board s
    let mutable Vs = Map.empty  //stores game.getValidMoves for board s

    member this.getActionProb canonicalBoard  temp =
        for i in 1..numMCTSSims do
            this.search canonicalBoard |> ignore
        ()

    member this.search(canonicalBoard:int list list):float =
        let s = game.stringRepresentation canonicalBoard 
        if not (Es.ContainsKey s) then
            Es <- Es.Add(s, game.gameEnded canonicalBoard 1)
        if Es.[s] <> 0.0 then
            //terminal node, get out
             -Es.[s]
        else if not (Ps.ContainsKey s) then
            //leaf node
            let mutable ps, v = nnet.predict 
            let valids = game.getValidMoves canonicalBoard 1
            ps <- List.map2 (fun x y -> x * y) ps valids
            //Ps[s] = self.Ps[s]*valids     //masking invalid moves

            //Ps[s] /= np.sum(self.Ps[s])   //renormalize
            let suma = List.sum ps
            ps <- List.map (fun x -> x/suma) ps

            Ps <- Ps.Add(s, ps)

            Vs <- Vs.Add(s, valids)
            Ns <- Ns.Add(s,  0)
            -v
        else
            0.0

type Coach(game) = 
    let game = game
    let nnet = NeuralNet(game)
    let pnet = NeuralNet(game)
    let mcts = MCTS(game, nnet)
    let trainExamplesHistory = [] //history of examples from args.numItersForTrainExamplesHistory latest iterations
    let skipFirstSelfPlay = false
    let mutable curPlayer = 0
    member this.executeEpisode() =
        let trainExamples = []
        let board = game.getInitBoard
        curPlayer <- 1
        let mutable episodeStep = 0
        while true do
            episodeStep <- episodeStep + 1
            let canonicalBoard = game.getCanonicalForm board curPlayer
            let temp = if episodeStep < tempThreshold then 1 else 0
            let p_i = mcts.getActionProb canonicalBoard temp
            ()

    member this.learn() = 
        for i = 1 to numIters + 1 do
            //# bookkeeping
            printfn "------ITER %d ------" i
            if not skipFirstSelfPlay || i > 1 then
                let mutable iterationTrainExamples = []
                for eps = 1 to numEps do 
                    let mcts = MCTS(game, nnet)
                    this.executeEpisode()
                    ()

        
[<EntryPoint>]
let main argv = 
    printfn "mumu"
    System.Diagnostics.Debug.WriteLine("mumu")
    let g = Game()
    let c = Coach(g)
    c.learn()

    10 // return an integer exit code
