from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel as BaseModel, ConfigDict, Field
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


metamodel_version = "None"
version = "None"

class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra='forbid',
        arbitrary_types_allowed=True,
        use_enum_values = True)


class NullDataOptions(str, Enum):
    
    
    UNSPECIFIED_METHOD_OF_ADMINISTRATION = "UNSPECIFIED_METHOD_OF_ADMINISTRATION"
    
    NOT_APPLICABLE = "NOT_APPLICABLE"
    
    NOT_MENTIONED = "NOT_MENTIONED"
    
    

class ExtractionResult(ConfiguredBaseModel):
    """
    A result of extracting knowledge on text
    """
    input_id: Optional[str] = Field(None)
    input_title: Optional[str] = Field(None)
    input_text: Optional[str] = Field(None)
    raw_completion_output: Optional[str] = Field(None)
    prompt: Optional[str] = Field(None)
    extracted_object: Optional[Any] = Field(None, description="""The complex objects extracted from the text""")
    named_entities: Optional[List[Any]] = Field(default_factory=list, description="""Named entities extracted from the text""")
    

class NamedEntity(ConfiguredBaseModel):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class EntityContainingDocument(NamedEntity):
    
    genes: Optional[List[str]] = Field(default_factory=list, description="""A semicolon-separated list of genes.""")
    proteins: Optional[List[str]] = Field(default_factory=list, description="""A semicolon-separated list of proteins.""")
    molecules: Optional[List[str]] = Field(default_factory=list, description="""A semicolon-separated list of molecules.""")
    organisms: Optional[List[str]] = Field(default_factory=list, description="""A semicolon-separated list of taxonomic terms of living things.""")
    gene_gene_interactions: Optional[List[GeneGeneInteraction]] = Field(default_factory=list, description="""A semicolon-separated list of gene-gene interactions.""")
    gene_protein_interactions: Optional[List[GeneProteinInteraction]] = Field(default_factory=list, description="""A semicolon-separated list of gene-protein interactions.""")
    gene_organism_relationships: Optional[List[GeneOrganismRelationship]] = Field(default_factory=list, description="""A semicolon-separated list of gene-organism relationships.""")
    protein_protein_interactions: Optional[List[ProteinProteinInteraction]] = Field(default_factory=list, description="""A semicolon-separated list of protein-protein interactions.""")
    protein_organism_relationships: Optional[List[ProteinOrganismRelationship]] = Field(default_factory=list, description="""A semicolon-separated list of protein-organism relationships.""")
    gene_molecule_interactions: Optional[List[GeneMoleculeInteraction]] = Field(default_factory=list, description="""A semicolon-separated list of gene-molecule interactions.""")
    protein_molecule_interactions: Optional[List[ProteinMoleculeInteraction]] = Field(default_factory=list, description="""A semicolon-separated list of protein-molecule interactions.""")
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class Gene(NamedEntity):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class Protein(NamedEntity):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class Molecule(NamedEntity):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class Organism(NamedEntity):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class CompoundExpression(ConfiguredBaseModel):
    
    None
    

class GeneGeneInteraction(CompoundExpression):
    
    gene1: Optional[str] = Field(None)
    gene2: Optional[str] = Field(None)
    

class GeneProteinInteraction(CompoundExpression):
    
    gene: Optional[str] = Field(None)
    protein: Optional[str] = Field(None)
    

class GeneOrganismRelationship(CompoundExpression):
    
    gene: Optional[str] = Field(None)
    organism: Optional[str] = Field(None)
    

class ProteinProteinInteraction(CompoundExpression):
    
    protein1: Optional[str] = Field(None)
    protein2: Optional[str] = Field(None)
    

class ProteinOrganismRelationship(CompoundExpression):
    
    protein: Optional[str] = Field(None)
    organism: Optional[str] = Field(None)
    

class GeneMoleculeInteraction(CompoundExpression):
    
    gene: Optional[str] = Field(None)
    molecule: Optional[str] = Field(None)
    

class ProteinMoleculeInteraction(CompoundExpression):
    
    protein: Optional[str] = Field(None)
    molecule: Optional[str] = Field(None)
    

class Triple(CompoundExpression):
    """
    Abstract parent for Relation Extraction tasks
    """
    subject: Optional[str] = Field(None)
    predicate: Optional[str] = Field(None)
    object: Optional[str] = Field(None)
    qualifier: Optional[str] = Field(None, description="""A qualifier for the statements, e.g. \"NOT\" for negation""")
    subject_qualifier: Optional[str] = Field(None, description="""An optional qualifier or modifier for the subject of the statement, e.g. \"high dose\" or \"intravenously administered\"""")
    object_qualifier: Optional[str] = Field(None, description="""An optional qualifier or modifier for the object of the statement, e.g. \"severe\" or \"with additional complications\"""")
    

class TextWithTriples(ConfiguredBaseModel):
    """
    A text containing one or more relations of the Triple type.
    """
    publication: Optional[Publication] = Field(None)
    triples: Optional[List[Triple]] = Field(default_factory=list)
    

class TextWithEntity(ConfiguredBaseModel):
    """
    A text containing one or more instances of a single type of entity.
    """
    publication: Optional[Publication] = Field(None)
    entities: Optional[List[str]] = Field(default_factory=list)
    

class RelationshipType(NamedEntity):
    
    id: str = Field(..., description="""A unique identifier for the named entity""")
    label: Optional[str] = Field(None, description="""The label (name) of the named thing""")
    

class Publication(ConfiguredBaseModel):
    
    id: Optional[str] = Field(None, description="""The publication identifier""")
    title: Optional[str] = Field(None, description="""The title of the publication""")
    abstract: Optional[str] = Field(None, description="""The abstract of the publication""")
    combined_text: Optional[str] = Field(None)
    full_text: Optional[str] = Field(None, description="""The full text of the publication""")
    

class AnnotatorResult(ConfiguredBaseModel):
    
    subject_text: Optional[str] = Field(None)
    object_id: Optional[str] = Field(None)
    object_text: Optional[str] = Field(None)
    


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
ExtractionResult.model_rebuild()
NamedEntity.model_rebuild()
EntityContainingDocument.model_rebuild()
Gene.model_rebuild()
Protein.model_rebuild()
Molecule.model_rebuild()
Organism.model_rebuild()
CompoundExpression.model_rebuild()
GeneGeneInteraction.model_rebuild()
GeneProteinInteraction.model_rebuild()
GeneOrganismRelationship.model_rebuild()
ProteinProteinInteraction.model_rebuild()
ProteinOrganismRelationship.model_rebuild()
GeneMoleculeInteraction.model_rebuild()
ProteinMoleculeInteraction.model_rebuild()
Triple.model_rebuild()
TextWithTriples.model_rebuild()
TextWithEntity.model_rebuild()
RelationshipType.model_rebuild()
Publication.model_rebuild()
AnnotatorResult.model_rebuild()

