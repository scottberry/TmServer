# TmServer - TissueMAPS server application.
# Copyright (C) 2016  Markus D. Herrmann, University of Zurich and Robin Hafen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""API view functions for querying :mod:`feature <tmlib.models.feature>`
resources.
"""
import csv
import json
import logging
import numpy as np
import pandas as pd
from cStringIO import StringIO
from flask_jwt import jwt_required
from flask import jsonify, request, send_file, Response, stream_with_context
from sqlalchemy import null
from sqlalchemy.orm.exc import NoResultFound

import tmlib.models as tm

from tmserver.api import api
from tmserver.util import (
    decode_query_ids, assert_query_params, assert_form_params,
    is_true, is_false
)
from tmserver.error import *


logger = logging.getLogger(__name__)


@api.route(
    '/experiments/<experiment_id>/features/<feature_id>',
    methods=['PUT']
)
@jwt_required()
@assert_form_params('name')
@decode_query_ids('read')
def update_feature(experiment_id, feature_id):
    """
    .. http:put:: /api/experiments/(string:experiment_id)/features/(string:feature_id)

        Update a :class:`Feature <tmlib.models.feature.Feature>`.

        **Example request**:

        .. sourcecode:: http

            Content-Type: application/json

            {
                "name": "New Name"
            }

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "message": "ok"
            }

        :statuscode 400: malformed request
        :statuscode 200: no error

    """
    data = request.get_json()
    name = data.get('name')
    logger.info('rename feature %d of experiment %d', feature_id, experiment_id)
    with tm.utils.ExperimentSession(experiment_id) as session:
        feature = session.query(tm.Feature).get(feature_id)
        feature.name = name
    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/features/<feature_id>', methods=['DELETE']
)
@jwt_required()
@decode_query_ids('write')
def delete_feature(experiment_id, feature_id):
    """
    .. http:delete:: /api/experiments/(string:experiment_id)/features/(string:feature_id)

        Delete a specific :class:`Feature <tmlib.models.feature.Feature>`.

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "message": "ok"
            }

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 401: not authorized

    """
    logger.info('delete feature %d of experiment %d', feature_id, experiment_id)
    with tm.utils.ExperimentSession(experiment_id, False) as session:
        # This query may take long in case there are many mapobjects and may
        # consequently cause uWSGI timeouts.
        session.query(tm.FeatureValue.values.delete(str(feature_id)))
        session.query(tm.Feature).filter_by(id=feature_id).delete()
    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/feature-values',
    methods=['POST']
)
@jwt_required()
@assert_form_params(
    'plate_name', 'well_name', 'tpoint', 'names', 'values', 'labels'
)
@decode_query_ids('write')
def add_feature_values(experiment_id, mapobject_type_id):
    """
    .. http:post:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/feature-values

        Add :class:`FeatureValues <tmlib.models.feature.FeatureValues>`
        for every :class:`Mapobject <tmlib.models.mapobject.Mapobject>` of the
        given :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>` at a
        given :class:`Site <tmlib.models.site.Site>` and time point.
        Feature values must be provided in form of a *n*x*p* array, where
        *n* are the number of objects (rows) and *p* the number of features
        (columns). Rows are identifiable by *labels* and columns by *names*.
        Provided *labels* must match the
        :attr:`label <tmlib.models.mapobject.MapobjectSegmentation.label>` of
        segmented objects.

        **Example request**:

        .. sourcecode:: http

            Content-Type: application/json

            {
                "plate_name": "plate1",
                "well_name": "D04",
                "well_pos_y": 0,
                "well_pos_x": 2,
                "tpoint": 0
                "names": ["feature1", "feature2", "feature3"],
                "labels": [1, 2],
                "values" [
                    [2.45, 8.83, 4.37],
                    [5.67, 7.21, 1.58]
                ]
            }

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request
        :statuscode 401: unauthorized
        :statuscode 404: not found

        :query names: names of features (required)
        :query labels: object labels (required)
        :query values: *m*x*n* array with *m* objects and *n* features (required)
        :query plate_name: name of the plate (required)
        :query well_name: name of the well (required)
        :query well_pos_x: x-coordinate of the site within the well (optional)
        :query well_pos_y: y-coordinate of the site within the well (optional)
        :query tpoint: time point (required)
        :query zplane: z-plane (required)

    """
    data = request.get_json()

    plate_name = data.get('plate_name')
    well_name = data.get('well_name')
    tpoint = int(data.get('tpoint'))
    well_pos_x = data.get('well_pos_x')
    well_pos_y = data.get('well_pos_y')
    if well_pos_y is not None and well_pos_x is not None:
        well_pos_y = int(well_pos_y)
        well_pos_x = int(well_pos_x)
    elif well_pos_y is not None or well_pos_x is None:
        raise MissingGETParameterError('well_pos_x')
    elif well_pos_y is None or well_pos_x is not None:
        raise MissingGETParameterError('well_pos_y')

    names = data.get('names')
    values = data.get('values')
    labels = data.get('labels')

    try:
        data = pd.DataFrame(values, columns=names, index=labels)
    except Exception as err:
        logger.error(
            'feature values were not provided in correct format: %s', str(err)
        )
        raise ResourceNotFoundError(
            'Feature values were not provided in the correct format.'
        )

    with tm.utils.ExperimentSession(experiment_id) as session:
        feature_lut = dict()
        for name in data.columns:
            feature = session.get_or_create(
                tm.Feature, name=name, mapobject_type_id=mapobject_type_id
            )
            feature_lut[name] = str(feature.id)
        data.rename(feature_lut, inplace=True)

    with tm.utils.ExperimentSession(experiment_id) as session:
        segmentation_layer = session.get_or_create(
            tm.SegmentationLayer,
            mapobject_type_id=mapobject_type_id, tpoint=tpoint, zplane=zplane
        )
        segmentation_layer_id = segmentation_layer.id

        well = session.query(tm.Well).\
            join(tm.Plate).\
            filter(tm.Plate.name == plate_name, tm.Well.name == well_name).\
            one()
        partition_key = well.id
        if well_pos_y is not None and well_pos_x is not None:
            site = session.query(tm.Site).\
                filter_by(y=well_pos_y, x=well_pos_x, well_id=well.id).\
                one()
            ref_instance = site
        else:
            ref_instance = well

        ref_mapobject_type = session.query(tm.MapobjectType).\
            filter_by(ref_type=ref_instance.__class__.__name__).\
            one()
        ref_segment = session.query(
                tm.MapobjectSegmentation.mapobject_id,
                tm.MapobjectSegmentation.geom_polygon
            ).\
            join(tm.Mapobject).\
            filter(
                tm.Mapobject.ref_id == ref_instance.id,
                tm.Mapobject.mapobject_type_id == ref_mapobject_type.id,
                tm.Mapobject.partition_key == partition_key
            ).\
            one()

        # This approach assumes that object segmentations have the same labels
        # across different z-planes.
        segmentations = session.query(
                tm.MapobjectSegmentation.label,
                tm.MapobjectSegmentation.mapobject_id,
            ).\
            filter(
                tm.MapobjectSegmentation.segmentation_layer_id == layer_id,
                tm.MapobjectSegmentation.partition_key == partition_key,
                tm.MapobjectSegmentation.geom_polygon.ST_CoveredBy(
                    ref_segment.geom_polygon
                )
            ).\
            all()
        if len(segmentations) == 0:
            raise ResourceNotFoundError(tm.MapobjectSegmentation)

        feature_values = list()
        for mapobject_id, label in segmentations:
            try:
                v = tm.FeatureValues(
                    partition_key=partition_key, mapobject_id=mapobject_id,
                    values=data.loc[label], tpoint=tpoint
                )
            except IndexError:
                raise ResourceNotFoundError(
                    tm.MapobjectSegmentation, label=label
                )
            feature_values.append(v)
        session.bulk_ingest(feature_values)

    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/feature-values',
    methods=['GET']
)
@jwt_required()
@decode_query_ids('read')
def get_feature_values(experiment_id, mapobject_type_id):
    """
    .. http:get:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/feature-values

        Get :class:`FeatureValues <tmlib.models.feature.FeatureValues>`
        for objects of the given
        :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>`
        in form of a *CSV* table with a row for each
        :class:`Mapobject <tmlib.models.mapobject.Mapobject>` and
        a column for each :class:`Feature <tmlib.models.feature.Feature>`.

        :query plate_name: name of the plate (optional)
        :query well_name: name of the well (optional)
        :query well_pos_x: x-coordinate of the site within the well (optional)
        :query well_pos_y: y-coordinate of the site within the well (optional)
        :query tpoint: time point (optional)

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request
        :statuscode 401: unauthorized
        :statuscode 404: not found

    .. note:: The table is send in form of a *CSV* stream with the first row
        representing column names.

    .. warning:: Feature values are only returned for mapobjects that are
        contained (fully enclosed!) by the specified region. In case mapobjects
        span multiple sites, one must not specify the `well_pos_y` or
        `well_pos_x` parameters.
    """
    plate_name = request.args.get('plate_name')
    well_name = request.args.get('well_name')
    well_pos_x = request.args.get('well_pos_x', type=int)
    well_pos_y = request.args.get('well_pos_y', type=int)
    tpoint = request.args.get('tpoint', type=int)

    with tm.utils.MainSession() as session:
        experiment = session.query(tm.ExperimentReference).get(experiment_id)
        experiment_name = experiment.name

    with tm.utils.ExperimentSession(experiment_id) as session:
        segmentation_layers = session.query(tm.SegmentationLayer).\
            filter_by(mapobject_type_id=mapobject_type_id)
        if tpoint is not None:
            segmentation_layers = segmentation_layers.filter_by(tpoint=tpoint)
        segmentation_layers = segmentation_layers.all()
        layer_ids = [s.id for s in segmentation_layers]

        mapobject_type = session.query(tm.MapobjectType).get(mapobject_type_id)
        if well_pos_y is not None or well_pos_x is not None:
            if mapobject_type.ref_type in {tm.Plate.__name__, tm.Well.__name__}:
                raise MalformedRequestError(
                    'Query parameters "well_pos_y" and "well_pos_x" are not '
                    'supported for mapobject type "%s"' % mapobject_type.name
                )
            ref_model_name = getattr(
                mapobject_type, 'ref_type', tm.Site.__name__
            )
            instances = session.query(
                    tm.Site.id.label('ref_id'),
                    tm.Site.well_id.label('partition_key'),
                ).\
                join(tm.Well).\
                join(tm.Plate)
        else:
            if well_name is not None:
                if mapobject_type.ref_type == tm.Plate.__name__:
                    raise MalformedRequestError(
                        'Query parameter "well_name" is not supported for '
                        'mapobject type "%s"' % mapobject_type.name
                    )
            ref_model_name = getattr(
                mapobject_type, 'ref_type', tm.Well.__name__
            )
            instances = session.query(
                    tm.Well.id.label('ref_id'),
                    tm.Well.id.label('partition_key'),
                ).\
                join(tm.Plate)

        filename_formatstring = '{experiment}'
        if plate_name is not None:
            filename_formatstring += '_{plate}'
            instances = instances.filter(tm.Plate.name == plate_name)
        if well_name is not None:
            filename_formatstring += '_{well}'
            instances = instances.filter(tm.Well.name == well_name)
        if well_pos_y is not None:
            filename_formatstring += '_y{y}'
            instances = instances.filter(tm.Site.y == well_pos_y)
        if well_pos_x is not None:
            filename_formatstring += '_x{x}'
            instances = instances.filter(tm.Site.x == well_pos_x)
        if tpoint is not None:
            filename_formatstring += '_t{t}'
        instances = instances.all()

        filename_formatstring += '_{object_type}_feature-values.csv'
        filename = filename_formatstring.format(
            experiment=experiment_name, plate=plate_name, well=well_name,
            y=well_pos_y, x=well_pos_x,
            t=tpoint, object_type=mapobject_type_name
        )

        features = session.query(tm.Feature.name).\
            filter_by(mapobject_type_id=mapobject_type_id).\
            order_by(tm.Feature.id).\
            all()
        feature_names = [f.name for f in features]

        ref_mapobject_type = session.query(tm.MapobjectType).\
            filter_by(ref_type=instances[0].__class__.__name__).\
            one()
        ref_mapobject_type_id = ref_mapobject_type.id

        instances = [(inst.ref_id, inst.partition_key) for inst in instances]

    def generate_feature_matrix():
        data = StringIO()
        w = csv.writer(data)

        column_names = ('id', ) + tuple(feature_names)
        w.writerow(column_names)
        yield data.getvalue()
        data.seek(0)
        data.truncate(0)

        for ref_id, partition_key in instances:
            with tm.utils.ExperimentSession(experiment_id) as session:
                ref_segment = session.query(tm.MapobjectSegmentation).\
                    join(tm.Mapobject).\
                    filter(
                        tm.Mapobject.ref_id == ref_id,
                        tm.Mapobject.mapobject_type_id == ref_mapobject_type_id,
                        tm.Mapobject.paritition_key == partition_key
                    ).\
                    one()
                # Use OUTER JOIN to also include mapobjects that don't have
                # any feature values.
                feature_values = session.query(
                        tm.MapobjectSegmentation.mapobject_id,
                        tm.FeatureValues.values
                    ).\
                    outerjoin(tm.FeatureValues).\
                    filter(
                        tm.FeatureValues.partition_key == partition_key,
                        tm.FeatureValues.tpoint.in_(tpoints),
                        tm.MapobjectSegmentation.segmentation_layer_id.in_(layer_ids),
                        tm.MapobjectSegmentation.geom_polygon.ST_CoveredBy(
                            ref_segment.geom_polygon
                        )
                    ).\
                    order_by(tm.MapobjectSegmentation.mapobject_id).\
                    all()

                for mapobject_id, vals in feature_values:
                    if vals is None:
                        logger.warn(
                            'no feature values found for mapobject %d',
                            mapobject_id
                        )
                        v = [str(np.nan) for x in range(len(feature_names))]
                    else:
                        # Values must be sorted based on feature_id, such that
                        # they end up in the correct column of the CSV table.
                        # Feature IDs must be sorted as integers to get the
                        # desired order.
                        # TODO: Can we use HSTORE slice upon SELECT to ensure
                        # values are in the correct order?
                        v = [vals[k] for k in sorted(vals, key=lambda k: int(k))]
                    v.insert(0, str(mapobject_id))
                    w.writerow(tuple(v))
                yield data.getvalue()
                data.seek(0)
                data.truncate(0)

    return Response(
        generate_feature_matrix(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename={filename}'.format(
                filename=filename
            )
        }
    )


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/metadata',
    methods=['GET']
)
@jwt_required()
@decode_query_ids('read')
def get_metadata(experiment_id, mapobject_type_id):
    """
    .. http:get:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/metadata

        Get positional information for
        the given :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>`
        as a *n*x*p* feature table, where *n* is the number of
        mapobjects (:class:`Mapobject <tmlib.models.mapobject.Mapobject>`) and
        *p* is the number of metadata attributes.

        :query plate_name: name of the plate (optional)
        :query well_name: name of the well (optional)
        :query well_pos_x: x-coordinate of the site within the well (optional)
        :query well_pos_y: y-coordinate of the site within the well (optional)
        :query tpoint: time point (optional)

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request
        :statuscode 401: unauthorized
        :statuscode 404: not found

    .. note:: The table is send in form of a *CSV* stream with the first row
        representing column names.

    .. warning:: Metadata are only returned for mapobjects that are
        contained (fully enclosed!) by the specified region. In case mapobjects
        span multiple sites, one must not specify the `well_pos_y` or
        `well_pos_x` parameters.
    """
    plate_name = request.args.get('plate_name')
    well_name = request.args.get('well_name')
    well_pos_x = request.args.get('well_pos_x', type=int)
    well_pos_y = request.args.get('well_pos_y', type=int)
    tpoint = request.args.get('tpoint', type=int)

    with tm.utils.MainSession() as session:
        experiment = session.query(tm.ExperimentReference).get(experiment_id)
        experiment_name = experiment.name

    with tm.utils.ExperimentSession(experiment_id) as session:
        segmentation_layers = session.query(tm.SegmentationLayer).\
            filter_by(mapobject_type_id=mapobject_type_id)
        if tpoint is not None:
            segmentation_layers = segmentation_layers.filter_by(tpoint=tpoint)
        segmentation_layers = segmentation_layers.all()
        layer_ids = [s.id for s in segmentation_layers]

        mapobject_type = session.query(tm.MapobjectType).get(mapobject_type_id)
        if well_pos_y is not None or well_pos_x is not None:
            if mapobject_type.ref_type in {tm.Plate.__name__, tm.Well.__name__}:
                raise MalformedRequestError(
                    'Query parameters "well_pos_y" and "well_pos_x" are not '
                    'supported for mapobject type "%s"' % mapobject_type.name
                )
            ref_model_name = getattr(
                mapobject_type, 'ref_type', tm.Site.__name__
            )
            instances = session.query(
                    tm.Site.id.label('ref_id'),
                    tm.Site.well_id.label('partition_key'),
                ).\
                join(tm.Well).\
                join(tm.Plate)
        else:
            if well_name is not None:
                if mapobject_type.ref_type == tm.Plate.__name__:
                    raise MalformedRequestError(
                        'Query parameter "well_name" is not supported for '
                        'mapobject type "%s"' % mapobject_type.name
                    )
            ref_model_name = getattr(
                mapobject_type, 'ref_type', tm.Well.__name__
            )
            instances = session.query(
                    tm.Well.id.label('ref_id'),
                    tm.Well.id.label('partition_key'),
                ).\
                join(tm.Plate)

        filename_formatstring = '{experiment}'
        if plate_name is not None:
            filename_formatstring += '_{plate}'
            instances = instances.filter(tm.Plate.name == plate_name)
        if well_name is not None:
            filename_formatstring += '_{well}'
            instances = instances.filter(tm.Well.name == well_name)
        if well_pos_y is not None:
            filename_formatstring += '_y{y}'
            instances = instances.filter(tm.Site.y == well_pos_y)
        if well_pos_x is not None:
            filename_formatstring += '_x{x}'
            instances = instances.filter(tm.Site.x == well_pos_x)
        if tpoint is not None:
            filename_formatstring += '_t{t}'
        instances = instances.all()

        filename_formatstring += '_{object_type}_metadata.csv'
        filename = filename_formatstring.format(
            experiment=experiment_name, plate=plate_name, well=well_name,
            y=well_pos_y, x=well_pos_x,
            t=tpoint, object_type=mapobject_type_name
        )

        results = session.query(tm.ToolResult.name).\
            filter_by(mapobject_type_id=mapobject_type_id).\
            order_by(tm.ToolResult.id).\
            all()
        result_names = [r.name for r in results]

        ref_mapobject_type = session.query(tm.MapobjectType).\
            filter_by(ref_type=instances[0].__class__.__name__).\
            one()
        ref_mapobject_type_id = ref_mapobject_type.id

        instances = [(inst.ref_id, inst.partition_key) for inst in instances]

    def generate_feature_matrix():
        data = StringIO()
        w = csv.writer(data)

        if ref_model_name == tm.Plate.__name__:
            position_names = [
                'plate_name'
            ]
        elif ref_model_name == tm.Well.__name__:
            position_names = [
                'plate_name', 'well_name'
            ]
        else:
            position_names = [
                'plate_name', 'well_name', 'well_pos_y', 'well_pos_x'
            ]
        segmentation_names = ['tpoint', 'label', 'is_border']

        column_names = ['id']
        column_names.extend(position_names)
        column_names.extend(segmentation_names)
        column_names.extend(tool_result_names)
        w.writerow(tuple(column_names))
        yield data.getvalue()
        data.seek(0)
        data.truncate(0)

        for ref_id, partition_key in instances:
            with tm.utils.ExperimentSession(experiment_id) as session:

                if ref_model_name == tm.Plate.__name__:
                    position_values = session.query(
                        tm.Plate.name.label('plate_name')
                        null().label('well_name'),
                        null().label('well_pos_y'),
                        null().label('well_pos_x')
                    ).\
                    join(tm.Well).\
                    filter(tm.Well.id == ref_id).\
                    one()
                elif ref_model_name == tm.Well.__name__:
                    position_values = session.query(
                        tm.Plate.name.label('plate_name'),
                        tm.Well.name.label('well_name'),
                        null().label('well_pos_y'),
                        null().label('well_pos_x')
                    ).\
                    filter(tm.Well.id == ref_id).\
                    one()
                else:
                    position_values = session.query(
                        tm.Plate.name.label('plate_name'),
                        tm.Well.name.label('well_name'),
                        tm.Site.y.label('well_pos_y'),
                        tm.Site.x.label('well_pos_x')
                    ).\
                    join(tm.Well).\
                    join(tm.Site).\
                    filter(tm.Site.id == ref_id).\
                    one()

                ref_segment = session.query(tm.MapobjectSegmentation).\
                    join(tm.Mapobject).\
                    filter(
                        tm.Mapobject.ref_id == ref_id,
                        tm.Mapobject.mapobject_type_id == ref_mapobject_type_id,
                        tm.Mapobject.paritition_key == partition_key
                    ).\
                    one()

                # LEFT OUTER JOIN to also include mapobjects that don't have
                # any feature values.
                is_border = tm.MapobjectSegmentation.geom_polygon.ST_Intersects(
                    ref_segment.geom_polygon.ST_Boundary()
                )
                label_values = session.query(
                        tm.MapobjectSegmentation.mapobject_id,
                        tm.LabelValues.values,
                        tm.LabelValues.tpoint,
                        tm.MapobjectSegmentation.label,
                        case([(is_border, True)], else_=False)
                    ).\
                    outerjoin(tm.LabelValues).\
                    filter(
                        tm.MapobjectSegmentation.segmentation_layer_id.in_(layer_ids),
                        tm.MapobjectSegmentation.geom_polygon.ST_CoveredBy(
                            ref_segment.geom_polygon
                        ),
                        tm.MapobjectSegmentation.partition_key == partition_key,
                        tm.LabelValues.tpoint.in_(tpoints)
                    ).\
                    order_by(tm.MapobjectSegmentation.mapobject_id).\
                    all()

                for mapobject_id, vals, tpoint, label, is_border in label_values:
                    if vals is None:
                        logger.warn(
                            'no label values found for mapobject %d '
                            'at time point %d', mapobject_id, tpoint
                        )
                        v = [str(np.nan) for x in range(len(tool_result_names))]
                    else:
                        # The order of keys in the HSTORE is not relevant and
                        # may not be reproduced on output. To ensure that
                        # values end up in the correct column, we sort values
                        # based on keys (feature IDs).
                        # Feature IDs must be sorted as integers to get the
                        # desired order.
                        v = [vals[k] for k in sorted(vals, key=lambda k: int(k))]
                    v.insert(0, str(mapobject_id))
                    v.extend(position_values)
                    v.extend([tpoint, label, is_border])
                    w.writerow(tuple(v))
                yield data.getvalue()
                data.seek(0)
                data.truncate(0)

    return Response(
        generate_feature_matrix(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename={filename}'.format(
                filename=filename
            )
        }
    )
