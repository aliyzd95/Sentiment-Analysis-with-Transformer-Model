"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import math
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

def hello_transformers():
    print("Hello from transformers_sentiment_analysis.py!")


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embeding dimension

        key: a Tensor of shape (K, M) where K is the sequence length and M is the 
            sequence embeding dimension

        value: a Tensor of shape (K, M) where K is the sequence length and M is 
            the sequence embeding dimension
             

    Returns
        out: a tensor of shape (K, M) which is the output of self-attention from
        the function
    """
    # make a placeholder for the output
    out = None
    ###############################################################################
    # TODO: Implement this function using exactly two for loops. For each of the  #
    # K queries, compute its dot product with each of the K keys. The scalar      #
    # output of the dot product will the be scaled by dividing it with the sqrt(M)#
    # Once we get all the K scaled weights corresponding to a query, we apply a   #
    # softmax function on them and use the value matrix to compute the weighted   #
    # sum of values using the matrix-vector product. This single vector computed  #
    # using weighted sum becomes an output to the Kth query vector                #
    ###############################################################################
    # Replace "pass" statement with your code
    
    K, M = query.shape  # طول دنباله و ابعاد بردارها را می‌گیریم
    _, M = key.shape  # ابعاد کلید را می‌گیریم
    QK = torch.zeros((K, K), device=query.device)  # ایجاد ماتریس خالی برای ضرب داخلی
    for i in range(K):  # برای هر کوئری
        for j in range(K):  # برای هر کلید
            QK[i, j] = torch.inner(query[i], key[j])  # ضرب داخلی کوئری و کلید
    QK = QK / (M**0.5)  # نرمال‌سازی با تقسیم بر ریشه ابعاد
    QK = torch.softmax(QK, dim=-1)  # محاسبه توزیع احتمال
    out = torch.mm(QK, value)  # محاسبه مجموع وزنی مقادیر

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the 
            sequence length and  M is the sequence embeding dimension

        key: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

        value: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

    Returns:
        out: a tensor of shape (N, K, M) that contains the weighted sum of values
           

    """
    # make a placeholder for the output
    out = None
    N, K, M = query.shape
    ###############################################################################
    # TODO: This function is extedning self_attention_two_loop_single for a batch #
    # of N. Implement this function using exactly two for loops. For each N       #
    # we have a query, key and value. The final output is the weighted sum of     #
    # values of these N queries and keys. The weight here is computed using scaled#
    # dot product  between each of the K queries and key. The scaling value here  #
    # is sqrt(M). For each of the N sequences, compute the softmaxed weights and  #
    # use them to compute weighted average of value matrix.                       #
    # Hint: look at torch.bmm                                                     #
    ###############################################################################
    # Replace "pass" statement with your code
    
    N, K, M = query.shape  # دریافت ابعاد ورودی
    QK = torch.zeros((N, K, K), device=query.device)  # ایجاد ماتریس برای ذخیره نتایج ضرب داخلی
    for i in range(K):  # برای هر طول دنباله
        for j in range(K):  # برای هر کلید
            QK[:, i, j] = torch.einsum('bi,bi->b', query[:, i], key[:, j])  # محاسبه ضرب داخلی
    QK /= (M ** 0.5)  # نرمال‌سازی با تقسیم بر ریشه ابعاد
    QK = torch.softmax(QK, dim=-1)  # محاسبه توزیع احتمال
    out = torch.bmm(QK, value)  # محاسبه مجموع وزنی مقادیر

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

        value: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension

             
        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values
              

    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values
           
        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    _, _, M = query.shape
    y = None
    weights_softmax = None
    ###############################################################################
    # TODO: This function performs same function as self_attention_two_loop_batch #
    # Implement this function using no loops.                                     #
    # For the mask part, you can ignore it for now and revisit it in the later part.
    # Given the shape of the mask is (N, K, K), and it is boolean with True values#
    # indicating  the weights that have to be masked and False values indicating  #
    # the weghts that dont need to be masked at that position. These masked-scaled#
    # weights can then be softmaxed to compute the final weighted sum of values   #
    # Hint: look at torch.bmm and torch.masked_fill                               #
    ###############################################################################
    # Replace "pass" statement with your code
    N,K,M=query.shape
    scaled=query.bmm(key.swapaxes(1,2))/(M**0.5)
    if mask is not None:
        ##########################################################################
        # TODO: Apply the mask to the weight matrix by assigning -1e9 to the     #
        # positions where the mask value is True, otherwise keep it as it is.    #
        ##########################################################################
        # Replace "pass" statement with your code
        scaled[mask==True]=-1e9
    
        # scaled = scaled.masked_fill(mask, -1e9)

    # Replace "pass" statement with your code
    
    N, K, M = query.shape  # دریافت ابعاد ورودی
    scaled = query.bmm(key.transpose(1, 2)) / (M ** 0.5)  # ضرب ماتریسی و نرمال‌سازی
    if mask is not None:  # اگر ماسک موجود باشد
        # جایگزینی مقادیر ماسک‌شده با -1e9 برای ممانعت از تاثیر در softmax
        scaled[mask] = -1e9  # scaled[mask == True] = -1e9

    weights_softmax = torch.softmax(scaled, dim=-1)  # محاسبه توزیع احتمال
    y = torch.bmm(weights_softmax, value)  # محاسبه مجموع وزنی مقادیر

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product_no_loop_batch to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and ley vector
            dim_v: an int value for output dimension for value vectors

        """
        self.q = None  # initialize for query
        self.k = None  # initialize for key
        self.v = None  # initialize for value
        self.weights_softmax = None
        ##########################################################################
        # TODO: This function initializes three functions to transform the 3 input
        # sequences to key, query and value vectors. More precisely, initialize  #
        # three nn.Linear layers that can transform the input with dimension     #
        # dim_in to query with dimension dim_q, key with dimension dim_q, and    #
        # values with dim_v. For each Linear layer, use the following strategy to#
        # initialize the weights:                                                #
        # If a Linear layer has input dimension D_in and output dimension D_out  #
        # then initialize the weights sampled from a uniform distribution bounded#
        # by [-c, c]                                                             #
        # where c = sqrt(6/(D_in + D_out))                                       #
        # Please use the same names for query, key and value transformations     #
        # as given above. self.q, self.k, and self.v respectively.               #
        ##########################################################################
        # Replace "pass" statement with your code
        
        # تعریف لایه‌های خطی برای کوئری، کلید و مقادیر
        self.q = nn.Linear(dim_in, dim_q)  # لایه خطی برای کوئری
        self.k = nn.Linear(dim_in, dim_q)  # لایه خطی برای کلید
        self.v = nn.Linear(dim_in, dim_v)  # لایه خطی برای مقادیر

        # مقداردهی اولیه وزن‌ها به صورت یکنواخت
        for layer in [self.q, self.k, self.v]:
            torch.nn.init.xavier_uniform_(layer.weight.data)  # مقداردهی اولیه وزن‌ها
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (K, M)
        return:
            y: Tensor of shape (N, K, dim_v)
        """
        self.weights_softmax = (
            None  # weight matrix after applying self_attention_no_loop_batch
        )
        y = None
        ##########################################################################
        # TODO: Use the functions initialized in the init fucntion to find the   #
        # output tensors. Precisely, pass the inputs query, key and value to the #
        #  three functions iniitalized above. Then, pass these three transformed #
        # query,  key and value tensors to the self_attention_no_loop_batch to   #
        # get the final output. For now, dont worry about the mask and just      #
        # pass it as a variable in self_attention_no_loop_batch. Assign the value#
        # of output weight matrix from self_attention_no_loop_batch to the       #
        # variable self.weights_softmax                                          #
        ##########################################################################
        # Replace "pass" statement with your code
        
        # عبور از ورودی‌ها از لایه‌های خطی و استفاده از تابع self_attention_no_loop_batch
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(
            self.q(query), self.k(key), self.v(value), mask
        )

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the batch size and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block



        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.


        """

        ##########################################################################
        # TODO: Initialize two things here:                                      #
        # 1.) Use nn.ModuleList to initialze a list of SingleHeadAttention layer #
        # modules.The length of this list should be equal to num_heads with each #
        # SingleHeadAttention layer having input dimension as dim_in, and query  #
        # , key, and value dimension as emb_out.                                 #
        # 2.) Use nn.Linear to map the output of nn.Modulelist block back to     #
        # dim_in                                                                 #
        ##########################################################################
        # Replace "pass" statement with your code
        
        self.num_heads = num_heads

        # ایجاد یک لیست از لایه‌های SelfAttention با تعداد سرها مشخص
        self.head = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])

        # لایه خطی برای تبدیل خروجی نهایی به dim_in
        self.linear = nn.Linear(num_heads * dim_out, dim_in)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)

        returns:
            y: Tensor of shape (N, K, M)
        """
        y = None
        ##########################################################################
        # TODO: You need to perform a forward pass through the MultiHeadAttention#
        # block using the variables defined in the initializing function. The    #
        # nn.ModuleList behaves as a list and you could use a for loop or list   #
        # comprehension to extract different elements of it. Each of the elements#
        # inside nn.ModuleList is a SingleHeadAttention that  will take the same #
        # query, key and value tensors and you will get a list of tensors as     #
        # output. Concatenate this list if tensors and pass them through the     #
        # nn.Linear mapping function defined in the initialization step.         #
        ##########################################################################
        # Replace "pass" statement with your code
        
        # انجام یک عبور جلو از طریق لایه‌های توجه چندگانه
        y = self.linear(torch.cat([h(query, key, value, mask) for h in self.head], dim=-1))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BathcNorm ,it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length and M is the sequence length embedding. We 
        compute the  mean with shape (N, K) and standard deviation with shape (N, K) 
        and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value

        """

        self.epsilon = epsilon

        ##########################################################################
        # TODO: Initialize the scale and shift parameters for LayerNorm.         #
        # Initialize the scale parameters to all ones and shift parameter to all #
        # zeros. As we have seen in the lecture, the shape of scale and shift    #
        # parameters remains the same as in Batchnorm, initialize these parameters
        # with appropriate dimensions                                            #
        ##########################################################################
        # Replace "pass" statement with your code
        
        self.scale = nn.Parameter(torch.ones(emb_dim))  # پارامتر مقیاس را با مقدار اولیه ۱
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # پارامتر جابجایی را با مقدار اولیه ۰

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x: Tensor):
        """
        An implementation of the forward pass of the Layer Normalization.

        args:
            x: a Tensor of shape (N, K, M) where N is the batch size, K is the
               sequence length and M is the embedding dimension
               
        returns:
            y: a Tensor of shape (N, K, M) after applying layer normalization
               
        """
        y = None
        ##########################################################################
        # TODO: Implement the forward pass of the LayerNormalization layer.      #
        # Compute the mean and standard deviation of input and use these to      #
        # normalize the input. Further, use self.gamma and self.beta to scale    #
        # these and shift this normalized input                                  #
        ##########################################################################
        # Replace "pass" statement with your code
        
        y = self.scale * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True, unbiased=False) + self.epsilon) + self.shift

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu -linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """

        ##########################################################################
        # TODO: initialize two MLPs here with the first one using inp_dim as input
        # dimension and hidden_dim_feedforward as output and the second with     #
        # hidden_dim_feedforward as input. You should figure out the output      #
        # dimesion of the second MLP.                                            #
        # HINT: Will the shape of input and output shape of the FeedForwardBlock #
        # change?                                                                #
        ##########################################################################
        # Replace "pass" statement with your code
        
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)  # لایه خطی اول با ورودی از ابعاد inp_dim و خروجی از ابعاد hidden_dim_feedforward
        torch.nn.init.xavier_uniform_(self.linear1.weight.data)  # مقداردهی اولیه وزن‌ها با توزیع یکنواخت به روش XAVIER

        self.r = nn.ReLU()  # تابع فعال‌سازی ReLU

        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)  # لایه خطی دوم با ورودی از ابعاد hidden_dim_feedforward و خروجی از ابعاد inp_dim
        torch.nn.init.xavier_uniform_(self.linear2.weight.data)  # مقداردهی اولیه وزن‌ها با توزیع یکنواخت به روش XAVIER

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of 
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        y = None
        ###########################################################################
        # TODO: Use the two MLP layers initialized in the init function to perform#
        # a forward pass. You should be using a ReLU layer after the first MLP and#
        # no activation after the second MLP                                      #
        ###########################################################################
        # Replace "pass" statement with your code
        
        # پیاده‌سازی تابع پیش‌خور در بلوک FeedForward

        y = self.linear1(x)  # عبور ورودی x از لایه خطی اول
        y = self.r(y)  # اعمال تابع فعال‌سازی ReLU بر روی خروجی لایه اول
        y = self.linear2(y)  # عبور خروجی از لایه خطی دوم

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        """
        This class implements the encoder block for the Transformer model, the 
        original paper used 6 of these blocks sequentially to train the final model. 
        Here, we will first initialize the required layers using the building  
        blocks we have already  implemented, and then finally write the forward     
        pass using these initialized layers, residual connections and dropouts.        
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of four components:
        
        1. MultiHead Attention
        2. FeedForward layer
        3. Residual connections after MultiHead Attention and feedforward layer
        4. LayerNorm
        
        The architecture is as follows:
        
       inp - multi_head_attention - out1 - layer_norm(out1 + inp) - dropout - out2 \ 
        - feedforward - out3 - layer_norm(out3 + out2) - dropout - out
        
        Here, inp is input of the MultiHead Attention of shape (N, K, M), out1, 
        out2 and out3 are the outputs of the corresponding layers and we add these 
        outputs to their respective inputs for implementing residual connections.

        args:
            num_heads: int value specifying the number of heads in the
                MultiHeadAttention block of the encoder

            emb_dim: int value specifying the embedding dimension of the input
                sequence

            feedforward_dim: int value specifying the number of hidden units in the 
                FeedForward layer of Transformer

            dropout: float value specifying the dropout value


        """

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        ##########################################################################
        # TODO: Initialize the following layers:                                 #
        # 1. One MultiHead Attention block using num_heads as number of heads and#
        #    emb_dim as the input dimension. You should also be able to compute  #
        #    the output dimension of MultiheadHead attention given num_heads and #
        #    emb_dim.                                                            #
        #    Hint: use the logic that you concatenate the output from each       #
        #    SingleHeadAttention inside the MultiHead Attention block and choose #
        #    the output dimension such that the concatenated tensor and the input#
        #    tensor have the same embedding dimension.                           #
        #                                                                        #
        # 2. Two LayerNorm layers with input dimension equal to emb_dim          #
        # 3. One feedForward block taking input as emb_dim and hidden units as   #
        #    feedforward_dim                                                     #
        # 4. A Dropout layer with given dropout parameter                        #
        ##########################################################################
        # Replace "pass" statement with your code
        
        self.MultiHeadBlock = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        # بلوک MultiHead Attention با تعداد سرها و ابعاد ورودی
        self.norm1 = LayerNormalization(emb_dim)  # لایه نرمال‌سازی اول
        self.norm2 = LayerNormalization(emb_dim)  # لایه نرمال‌سازی دوم
        self.ffn = FeedForwardBlock(emb_dim, feedforward_dim)  # بلوک FeedForward
        self.dropout = nn.Dropout(dropout)  # لایه Dropout
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x):

        """

        An implementation of the forward pass of the EncoderBlock of the
        Transformer model.
        args:
            x: a Tensor of shape (N, K, M) as input sequence
        returns:
            y: a Tensor of shape (N, K, M) as the output of the forward pass
        """
        y = None
        ##########################################################################
        # TODO: Use the layer initialized in the init function to complete the   #
        # forward pass. As Multihead Attention takes in 3 inputs, use the same   #
        # input thrice as the input. Follow the Figure 1 in Attention is All you #
        # Need paper to complete the rest of the forward pass. You can also take #
        # reference from the architecture written in the fucntion documentation. #
        ##########################################################################
        # Replace "pass" statement with your code
        
        x1 = self.dropout(self.norm1(x + self.MultiHeadBlock(x, x, x)))
        # x1: خروجی مرحله اول که شامل نرمال‌سازی و Dropout است.
        y = self.dropout(self.norm2(x1 + self.ffn(x1)))
        # y: خروجی نهایی که از مرحله دوم نرمال‌سازی و Dropout بدست آمده است.

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y



class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        The class encapsulates the implementation of the final Encoder that use
        multiple EncoderBlock layers.

        args:
            num_heads: int representing number of heads to be used in the
                EncoderBlock
            emb_dim: int repreesenting embedding dimension for the Transformer
                model
            feedforward_dim: int representing hidden layer dimension for the
                feed forward block

        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq





def position_encoding_simple(K: int, M: int) -> Tensor:
    """
    An implementation of the simple positional encoding using uniform intervals
    for a sequence.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence
           
    return:
        y: a Tensor of shape (1, K, M)
    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K, construct a 1D Tensor of length#
    # K with nth element as n/K, where n starts from 0. Replicate this tensor M  #
    # times to create a tensor of the required output shape                      #
    ##############################################################################
    # Replace "pass" statement with your code
    
    y = torch.linspace(0, 1 - 1 / K, steps=K)
    # ایجاد یک تنسور 1D با طول K که شامل مقادیر یکنواخت از 0 تا 1 است.
    
    y = y.repeat((1, M, 1)).permute(0, 2, 1)
    # تکرار این تنسور M بار و تغییر شکل آن به ابعاد (1، K، M) برای مطابقت با خروجی مورد نیاز.


    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return y



def position_encoding_sinusoid(K: int, M: int) -> Tensor:

    """
    An implementation of the sinousoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence
           
    return:
        y: a Tensor of shape (1, K, M)

    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K and embedding dimension M       #
    # construct a tesnor of shape (K, M) where the value along the dimensions    #
    # follow the equations given in the notebook. Make sure to keep in mind the  #
    # alternating sines and cosines along the embedding dimension M.             #
    ##############################################################################
    # Replace "pass" statement with your code
    
    temp=torch.arange(M).reshape(1,M)
    a = 2*torch.floor(temp / M)
    p = torch.arange(K).reshape(K,1)

    y = torch.zeros(1,K, M)
    y[:,:, 0::2] = torch.sin(p / torch.pow(10000, a[0, 0::2]))
    y[:,:, 1::2] = torch.cos(p / torch.pow(10000, a[0, 1::2]))
    return y


class Transformer_encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        vocab_len: int,
        n_classes: int
    ):
        super().__init__()

        self.emb_layer = nn.Embedding(vocab_len,emb_dim)

        # Replace "pass" statement with your code
        ##############################################################################
        
        # لایهٔ Embedding برای تبدیل ورودی‌ها به بردارهای با ابعاد emb_dim
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)

        # ایجاد لیستی از لایه‌های EncoderBlock
        self.enc_layers = nn.ModuleList([
            EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout) 
            for _ in range(num_enc_layers)
        ])

        # لایه نهایی برای پیش‌بینی کلاس‌ها
        
        self.fc_out = nn.Linear(emb_dim, n_classes)
        
        ##############################################################################
        


    def forward(self, ques_b) -> Tensor:
      
    # Replace "pass" statement with your code
    ##############################################################################
        # مرحلهٔ اول: تبدیل ورودی به نمایه‌های emb_dim
        x = self.emb_layer(ques_b)

        # مرحلهٔ دوم: اضافه کردن موقعیت‌گذاری به ورودی
        pos_enc = position_encoding_simple(x.size(1), x.size(2)).to(x.device)  # K و M
        x += pos_enc

        # مرحلهٔ سوم: عبور از لایه‌های EncoderBlock
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        # مرحلهٔ چهارم: گرفتن نمای نهایی و پیش‌بینی کلاس
        # انتخاب پیش‌بینی مربوط به آخرین توکن
        x = x[:, -1, :]  # انتخاب آخرین توکن
        y = self.fc_out(x)  # پیش‌بینی کلاس

        return y

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################




