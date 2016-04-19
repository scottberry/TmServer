import numpy as np

from sqlalchemy import Integer, ForeignKey, Column, String
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON

from tmaps.serialize import json_encoder
from tmaps.extensions import db
from tmaps.model import Model


class LabelResult(Model):
    __tablename__ = 'result_labelresults'

    result_type = Column(String)
    attributes = Column(JSON)
    tool_session_id = \
        Column(Integer, ForeignKey('tool_sessions.id'))
    mapobject_type_id = Column(
        Integer, ForeignKey('mapobject_types.id'))
    mapobject_type = relationship(
        'MapobjectType', backref='label_results')

    def __init__(self, ids, labels, mapobject_type, session, attributes=None):
        self.result_type = self.__class__.__name__
        self.mapobject_type_id = mapobject_type.id
        self.tool_session_id = session.id
        self.attributes = attributes

        db.session.add(self)
        db.session.flush()

        label_objs = []
        for mapobject_id, label in zip(ids, labels):
            pl = LabelResultLabel(
                mapobject_id=mapobject_id, label=label,
                label_result_id=self.id)
            label_objs.append(pl)

        db.session.add_all(label_objs)
        db.session.commit()

    def get_labels_for_objects(self, mapobject_ids):
        return dict(
            [(l.mapobject_id, l.label)
             for l in self.labels
             if l.mapobject_id in set(mapobject_ids)])


@json_encoder(LabelResult)
def encode_tool(obj, encoder):
    return {
        'id': obj.hash,
        'result_type': obj.result_type,
        'attributes': obj.attributes
    }


class LabelResultLabel(Model):
    __tablename__ = 'result_labelresult_labels'

    mapobject_id = Column(
        Integer, ForeignKey('mapobjects.id'))
    label_result_id = \
        Column(Integer, ForeignKey('result_labelresults.id'))
    label = Column(JSON)

    label_result = relationship('LabelResult', backref='labels')
    mapobject = relationship('Mapobject', backref='labels')


class ScalarLabelResult(LabelResult):
    def __init__(self, ids, labels, mapobject_type, session, attributes={}):
        attributes.update({
            'labels': list(set(labels))
        })
        super(ScalarLabelResult, self).__init__(
            ids, labels, mapobject_type, session, attributes=attributes)


class ContinuousLabelResult(LabelResult):
    def __init__(self, ids, labels, mapobject_type, session, attributes={}):
        attributes.update({
            'min': np.min(labels),
            'max': np.max(labels)
        })
        super(ContinuousLabelResult, self).__init__(
            ids, labels, mapobject_type, session, attributes=attributes)