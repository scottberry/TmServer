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
"""API view functions for querying :mod:`mapobject <tmlib.models.mapobject>` 
resources.
"""
import json
import logging
import numpy as np
import pandas as pd
from cStringIO import StringIO
from flask_jwt import jwt_required
from flask import jsonify, request, send_file, Response
from sqlalchemy.orm.exc import NoResultFound
from werkzeug import secure_filename

import tmlib.models as tm
from tmlib.image import SegmentationImage
from tmlib.metadata import SegmentationImageMetadata

from tmserver.api import api
from tmserver.util import (
    decode_query_ids, assert_query_params, assert_form_params,
    is_true, is_false
)
from tmserver.error import *


logger = logging.getLogger(__name__)


@api.route('/experiments/<experiment_id>/mapobject_types', methods=['GET'])
@jwt_required()
@decode_query_ids('read')
def get_mapobject_types(experiment_id):
    """
    .. http:get:: /api/experiments/(string:experiment_id)/mapobject_types

        Get the supported mapobject types for a specific experiment.

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "data": [
                    {
                        "id": "MQ==",
                        "name": "Cells",
                        "features": [
                            {
                                "id": "MQ==",
                                "name": "Cell_Area"
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }

        :query name: name of a mapobject type (optional)

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error

    """
    logger.info('get all mapobject types from experiment %d', experiment_id)
    name = request.args.get('name')
    with tm.utils.ExperimentSession(experiment_id) as session:
        mapobject_types = session.query(tm.MapobjectType)
        if name is not None:
            logger.info('filter mapobject types by name "%s"', name)
            mapobject_types = mapobject_types.filter_by(name=name)
        mapobject_types = mapobject_types.\
            order_by(tm.MapobjectType.name).\
            all()
        return jsonify(data=mapobject_types)


@api.route(
    '/experiments/<experiment_id>/mapobject_types', methods=['POST']
)
@jwt_required()
@assert_form_params('name')
@decode_query_ids('write')
def create_mapobject_type(experiment_id):
    """
    .. http:post:: /api/experiments/(string:experiment_id)/mapobject_type

        Create a :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>`

        **Example request**:

        .. sourcecode:: http

            Content-Type: application/json

            {
                "name": "Cells"
            }

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "data": {
                    "id": "MQ==",
                    "name": "Cells",
                    "features": [
                        {
                            "id": "MQ==",
                            "name": "Cell_Area"
                        },
                        ...
                    ]
                }
            }

        :statuscode 400: malformed request
        :statuscode 200: no error

    """
    data = request.get_json()
    name = data.get('name')
    logger.info(
        'create mapobject type "%s" for experiment %d', name, experiment_id
    )
    with tm.utils.ExperimentSession(experiment_id) as session:
        mapobject_type = session.get_or_create(
            tm.MapobjectType, name=name, experiment_id=experiment_id,
            ref_type=tm.Site.__name__
        )
        return jsonify(data=mapobject_type)


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>',
    methods=['PUT']
)
@jwt_required()
@assert_form_params('name')
@decode_query_ids('write')
def update_mapobject_type(experiment_id, mapobject_type_id):
    """
    .. http:put:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)

        Update a :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>`

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
    logger.info(
        'rename mapobject type %d of experiment %d',
        mapobject_type_id, experiment_id
    )
    with tm.utils.ExperimentSession(experiment_id) as session:
        mapobject_type = session.query(tm.MapobjectType).\
            get(mapobject_type_id)
        mapobject_type.name = name
    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>',
    methods=['DELETE']
)
@jwt_required()
@decode_query_ids('write')
def delete_mapobject_type(experiment_id, mapobject_type_id):
    """
    .. http:delete:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)

        Delete a specific :class:`MapobjectType <tmlib.models.mapobject.MapobjectType>`.

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
    logger.info(
        'delete mapobject type %d of experiment %d',
        mapobject_type_id, experiment_id
    )
    with tm.utils.ExperimentSession(experiment_id, False) as session:
        # This may take a long time and potentially cause problems with uWSGI
        # timeouts.
        session.query(tm.Mapobject).\
            filter_by(mapobject_type_id=mapobject_type_id).\
            delete()
        session.query(tm.MapobjectType).\
            filter_by(id=mapobject_type_id).\
            delete()
    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/features',
    methods=['GET']
)
@jwt_required()
@decode_query_ids('read')
def get_features(experiment_id, mapobject_type_id):
    """
    .. http:get:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/features

        Get a list of feature objects supported for this experiment.

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "data": [
                    {
                        "id": "MQ==",
                        "name": "Morpholgy_Area"
                    },
                    ...
                ]
            }

        :query name: name of a feature (optional)

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request

    """
    logger.info(
        'get features for experiment %d and mapobject type %d',
        experiment_id, mapobject_type_id
    )
    name = request.args.get('name')
    with tm.utils.ExperimentSession(experiment_id) as session:
        features = session.query(tm.Feature).\
            filter_by(mapobject_type_id=mapobject_type_id)
        if name is not None:
            logger.info('filter features by name "%s"', name)
            features = features.filter_by(name=name)
        features = features.order_by(tm.Feature.name).all()
        if not features:
            logger.waring(
                'no features found for mapobject type %d', mapobject_type_id
            )
        return jsonify(data=features)


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/segmentations',
    methods=['POST']
)
@jwt_required()
@assert_form_params(
    'plate_name', 'well_name', 'zplane', 'tpoint', 'image'
)
@decode_query_ids('write')
def add_segmentations(experiment_id, mapobject_type_id):
    """
    .. http:post:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/segmentations

        Provide segmentations in form of a labeled 2D pixels array
        for a given :class:`Site <tmlib.models.site.Site>`.
        A :class:`Mapobject <tmlib.models.mapobject.Mapobject>` and
        :class:`MapobjectSegmentation <tmlib.models.mapobject.MapobjectSegmentation>`
        will be created for each labeled connected pixel component in *image*.

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request

        :query image: 2D pixels array (required)
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
    zplane = int(data.get('zplane'))
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

    logger.info(
        'add segmentations for mapobject type %d of experiment %d at '
        'plate "%s", well "%s", zplane %d, time point %d',
        mapobject_type_id, experiment_id, plate_name, well_name, zplane, tpoint
    )

    pixels = data.get('image')
    array = np.array(pixels, dtype=np.int32)
    labels = np.unique(array[array > 0])
    n_objects = len(labels)

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

        y_offset, x_offset = ref_instance.offset
        if array.shape != ref_instance.image_size:
            raise MalformedRequestError('Provided image has wrong dimensions.')

        image = SegmentationImage(array)

        existing_segmentations_map = dict(
            session.query(
                tm.MapobjectSegmentation.label,
                tm.MapobjectSegmentation.mapobject_id
            ).\
            filter(
                tm.MapobjectSegmentation.partition_key == partition_key,
                tm.MapobjectSegmentation.segmentation_layer_id == layer_id,
                tm.MapobjectSegmentation.label.in_(labels),
                tm.MapobjectSegmentation.geom_centroid.ST_CoveredBy(
                    ref_segment.geom_polygon
                )
            ).\
            all()
        )

    with tm.utils.ExperimentSession(experiment_id, False) as session:
        segmentations = list()
        for label, polygon in image.extract_polygons(y_offset, x_offset):
            if label in existing_segmentations_map:
                # A parent mapobject with the same label may already exist,
                # since it may have been created for another zplane/tpoint.
                mapobject_id = existing_segmentations_map[label]
                mapobject = tm.Mapobject(partition_key, mapobject_type_id)
            else:
                session.add(mapobject)
                session.flush()
            segmentation = tm.MapobjectSegmentation(
                partition_key=paritition_key, mapobject_id=mapobject.id,
                geom_polygon=polygon, geom_centroid=polygon.centroid,
                segmentation_layer_id=segmentation_layer_id, label=label
            )
            session.add(segmentation)

    return jsonify(message='ok')


@api.route(
    '/experiments/<experiment_id>/mapobject_types/<mapobject_type_id>/segmentations',
    methods=['GET']
)
@jwt_required()
@assert_query_params(
    'plate_name', 'well_name', 'zplane', 'tpoint'
)
@decode_query_ids('read')
def get_segmentations(experiment_id, mapobject_type_id):
    """
    .. http:get:: /api/experiments/(string:experiment_id)/mapobject_types/(string:mapobject_type_id)/segmentations

        Get segmentations for each
        :class:`Mapobject <tmlib.model.mapobject.Mapobject>` contained within
        the specified :class:`Site <tmlib.models.site.Site>`. Segmentations are
        provided in form of a 2D labeled array, where pixel values encode
        object identity with unsigned integer values and background pixels are
        zero.

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Content-Type: application/json

            {
                "data": [
                   [1205, 7042, 4438, 7446, 3213, 8773, 5445, 9884, 8326, 6357],
                   [4663, 2740, 9954, 5187,  309, 8029, 4502, 4927, 5259, 1802],
                   [8117, 8489, 8495, 1194, 9788, 8182, 5431, 9969, 5931, 6490],
                   [7974, 3990, 8892, 1866, 7890, 1147, 9630, 3687, 1565, 3613],
                   [3977, 7318, 5252, 3270, 6746,  822, 7035, 5184, 7808, 4013],
                   [4380, 6719, 5882, 4279, 7996, 2139, 1760, 2548, 3753, 5511],
                   [7829, 8825,  224, 1192, 9296, 1663, 5213, 9040,  463, 9080],
                   [6922, 6781, 9776, 9002, 6992, 8887, 9672, 8500, 1085,  563]
                ]
            }

        :reqheader Authorization: JWT token issued by the server
        :statuscode 200: no error
        :statuscode 400: malformed request

        :query plate_name: name of the plate (required)
        :query well_name: name of the well (required)
        :query well_pos_x: x-coordinate of the site within the well (optional)
        :query well_pos_y: y-coordinate of the site within the well (optional)
        :query tpoint: time point (required)
        :query zplane: z-plane (required)

    """
    plate_name = request.args.get('plate_name')
    well_name = request.args.get('well_name')
    well_pos_x = request.args.get('well_pos_x', type=int)
    well_pos_y = request.args.get('well_pos_y', type=int)
    zplane = request.args.get('zplane', type=int)
    tpoint = request.args.get('tpoint', type=int)

    logger.info(
        'get segmentations for mapobject type %d of experiment %d at '
        'plate "%s", well "%s", zplane %d, time point %d',
        mapobject_type_id, experiment_id, plate_name, well_name, zplane, tpoint
    )

    with tm.utils.MainSession() as session:
        experiment = session.query(tm.ExperimentReference).get(experiment_id)
        experiment_name = experiment.name

    with tm.utils.ExperimentSession(experiment_id) as session:
        segmentation_layer = session.get_or_create(
            tm.SegmentationLayer,
            mapobject_type_id=mapobject_type_id, tpoint=tpoint, zplane=zplane
        )
        layer_id = segmentation_layer.id

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

        y_offset, x_offset = ref_instance.offset
        height, width = ref_instance.height, ref_instance.width

        polygons = session.query(
                tm.MapobjectSegmentation.label,
                tm.MapobjectSegmentation.geom_polygon
            ).\
            filter(
                tm.MapobjectSegmentation.partition_key == partition_key,
                tm.MapobjectSegmentation.segmentation_layer_id == layer_id,
                tm.MapobjectSegmentation.geom_polygon.ST_CoveredBy(
                    ref_segment.geom_polygon
                )
            ).\
            all()

        if len(polygons) == 0:
            raise ResourceNotFoundError(tm.MapobjectSegmentation, request.args)

    img = SegmentationImage.create_from_polygons(
        polygons, y_offset, x_offset, (height, width)
    )
    return jsonify(data=img.array.tolist())


