#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "DenseEmbeddingRequest",
    "DenseEmbeddingResponse",
    "SparseEmbeddingRequest",
    "QueryEncodingRequest",
    "DocumentEncodingRequest",
    "SparseEmbeddingResponse",
    "RerankingRequest",
    "RerankingResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "InsertRequest",
    "InsertResponse",
    "DeleteRequest",
    "DeleteResponse",
]

from typing import Dict, List, Optional, Union

from pydantic import ConfigDict, Field, model_validator

from agent_types.common import CSRArray, NDArray, Request, Response


class DenseEmbeddingRequest(Request):
    """稠密向量表征请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "dense_encode"

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )
    normalize: bool = Field(
        title="是否规一化",
        description="是否对结果向量进行规一化",
        default=True
    )


class DenseEmbeddingResponse(Response):
    """稠密向量表征响应"""

    model_config = ConfigDict(extra="allow")

    embedding: NDArray = Field(
        title="表征向量",
        description="稠密表征向量（若表征多条文本，则返回矩阵）"
    )


class SparseEmbeddingRequest(Request):
    """稀疏向量编码"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "sparse_encode"

    text: Union[str, List[str]] = Field(
        title="文本",
        description="需要表征的文本"
    )


class QueryEncodingRequest(SparseEmbeddingRequest):
    """面向Query的稀疏向量编码"""

    __request_name__ = "encode_queries"


class DocumentEncodingRequest(SparseEmbeddingRequest):
    """面向Document的稀疏向量编码"""

    __request_name__ = "encode_documents"


class SparseEmbeddingResponse(Response):
    """稀疏向量编码"""

    model_config = ConfigDict(extra="allow")

    embedding: CSRArray = Field(
        title="表征向量",
        description="稀疏表征向量（若表征多条文本，则返回矩阵）"
    )


class RerankingRequest(Request):
    """重排序请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "rerank"

    query: str = Field(
        title="用户查询",
        description="检索所需的用户查询"
    )
    documents: List[str] = Field(
        title="用户查询检索数据集",
        description="用户查询检索数据集(粗排结果)"
    )
    top_k: int = Field(
        title="重排序检索返回Top-k",
        description="重排序检索返回Top-k",
        default=6,
        ge=1,
        le=50
    )
    return_scores: bool = Field(
        title="是否返回分数",
        description="是否返回分数",
        default=True
    )
    return_documents: bool = Field(
        title="是否返回召回文档",
        description="是否返回召回文档",
        default=False
    )


class RerankingResponse(Response):
    """重排序响应"""

    model_config = ConfigDict(extra="allow")

    ranked: List[int] = Field(
        title="重排序检索返回Top-K索引",
        description="重排序检索返回Top-K索引",
        default_factory=list
    )
    scores: List[float] = Field(
        title="重排序检索返回Top-K分数",
        description="重排序检索返回Top-K索引",
        default_factory=list
    )
    documents: Optional[List[str]] = Field(
        title="召回文档",
        description="召回文档",
        default=None
    )


class RetrievalRequest(Request):
    """知识库检索请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "retrieve"

    collections: List[str] = Field(
        title="检索集合名称",
        description="检索集合名称，多个取值表示同时从多个集合中检索"
    )
    query: str = Field(
        title="用户查询",
        description="检索所需的用户查询"
    )
    top_k: int = Field(
        title="检索返回Top-k",
        description="检索返回Top-k",
        default=6,
        ge=1,
        le=50
    )
    expr: Dict[str, str] = Field(
        title="元数据表达式",
        description="元数据表达式，即筛选条件",
        default_factory=dict
    )
    index_fields: List[str] = Field(
        title="索引字段名称",
        description="索引字段的名称，默认为vector，可以指定多个表示混合检索",
        default_factory=lambda: ["vector"]
    )
    output_fields: Optional[List[str]] = Field(
        title="返回字段",
        description="检索结果中需包含的字段列表, 默认返回除向量字段以外的所有字段",
        default=None
    )
    use_rerank: bool = Field(
        title="是否使用Rerank",
        description="是否使用Rerank",
        default=True
    )
    pre_top_k: Optional[int] = Field(
        title="Rerank候选集大小",
        description="Rerank候选集大小",
        default=None,
        ge=1,
        le=50
    )
    search_kw: Optional[Dict] = Field(
        title="可选检索参数",
        description="可选检索参数(`keywords`, `timeout`, ...)",
        default_factory=dict,
    )


class RetrievalResponse(Response):
    """知识库检索响应"""

    model_config = ConfigDict(extra="allow")

    distance: List[float] = Field(
        title="检索距离",
        description="检索距离",
        default_factory=list
    )
    scores: List[float] = Field(
        title="重排分数",
        description="重排分数",
        default_factory=list
    )
    items: List[Dict] = Field(
        title="检索对象列表",
        description="检索对象列表",
        default_factory=list
    )
    retrieval_info: Optional[Dict] = Field(
        title="Extra检索消息详情",
        description="Extra检索消息详情(`timing`, `processing`, ...)",
        default_factory=dict
    )


class InsertRequest(Request):
    """插入知识样本"""

    __request_name__ = "insert"

    collection: str = Field(
        title="表名",
        description="若表不存在则会自动执行创建，其表结构由插入的第一个样本的属性名称及类型推导而来"
    )
    documents: List[Dict] = Field(
        title="插入数据集",
        default_factory=list,
        max_length=1024
    )
    indexed_field: str = Field(
        title="数据集索引列[Key]",
        default="text"
    )
    indexed_related_field: Optional[str] = Field(
        title="数据集索引列[Value]",
        default=None,
    )

    @model_validator(mode="after")
    def _model_validator_value(self):
        self.indexed_related_field = self.indexed_related_field or self.indexed_field
        return self


class InsertResponse(Response):
    """插入知识样本"""

    insert_msg: str = Field(
        title="插入数据消息",
        default=""
    )
    insert_count: int = Field(
        title="插入数据集长度",
        default=-1
    )


class DeleteRequest(Request):
    """删除样本"""

    __request_name__ = "delete"

    collection: str = Field(
        title="Milvus Collection Name"
    )
    expr: Optional[str] = Field(
        title="基于元数据表达式删除",
        default=None
    )
    delete_primary_ids: List = Field(
        title="基于ID删除",
        default_factory=list
    )

    @model_validator(mode="after")
    def _model_validator_value(self):
        ids = self.delete_primary_ids
        if ids:
            e1 = f"_id in {ids}"  # TODO, fixed,
            expr = e1 if not self.expr else f"({self.expr}) or ({e1})"
            self.expr = expr
        return self


class DeleteResponse(Response):
    """删除样本"""

    delete_count: int = Field(
        title="删除数据集长度",
        default=-1
    )
