#! /usr/bin/env python
#
"""
A minimal web application for checking the status of a GC3Pie session.

Provides a REST API to perform the basic GC3Pie operations on jobs,
plus a status page reporting some basic metrics.

It is implemented as a `Flask <http://flask.pocoo.org/>` "blueprint"
for easier embedding into larger web applications.

"""
# Copyright (C) 2015 S3IT, University of Zurich.
#
# Authors:
#   Riccardo Murri <riccardo.murri@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import absolute_import

__docformat__ = 'reStructuredText'
__version__ = '$Revision$'


# 3rd party imports
from flask import current_app
from flask import _app_ctx_stack as stack

import gc3libs
from tmlib import engine as tmlib_engine


class GC3Pie(object):
    def __init__(self, session, engine):
        self.session = session
        self.engine = engine


class GC3PieEngine(object):
    """
    A flask extension to perform the core GC3Pie operations on tasks
    in the given `session`.

    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            self.init_app(*args, **kwargs)

    def init_app(self, app):
        """Construct an `GC3PieEngine` Flask extension object."""

        if 'GC3PIE_SESSION_DIR' not in app.config or \
                'SQLALCHEMY_DATABASE_URI' not in app.config:
            raise ValueError(
                'GC3Pie extension needs values for GC3PIE_SESSION_DIR '
                'and SQLALCHEMY_DATABASE_URI'
            )

        # Gc3pie expects URIs pointing to postgres databases
        # to start with postgres:// instead of postgresql://.
        gc3pie_store_uri = \
            app.config['SQLALCHEMY_DATABASE_URI'].\
            replace('postgresql', 'postgres')
        gc3pie_session_dir = app.config.get('GC3PIE_SESSION_DIR')

        session = self._create_session(gc3pie_store_uri, gc3pie_session_dir)
        engine = self._create_bg_engine()
        # Save the session and engine objects globally on this flask application
        app.extensions['gc3pie'] = GC3Pie(session, engine)

        # Add existing tasks
        for task in session:
            engine.add(task)

    def _create_bg_engine(self):
        """Create and return a `tmlib.BgEngine`:class: instance."""
        engine = gc3libs.create_engine()
        bg_engine = tmlib_engine.BgEngine('threading', engine)
        return bg_engine

    def _create_session(self, gc3pie_store_uri, gc3pie_session_dir):
        """Create a sql-backed gc3pie session."""
        session = gc3libs.session.Session(
            gc3pie_session_dir,
            store_url=gc3pie_store_uri,
            table_name='gc3pie_tasks'
        )
        return session

    @property
    def session(self):
        return current_app.extensions['gc3pie'].session
        # ctx = stack.top
        # if ctx is not None:
        #     if not hasattr(ctx, 'gc3pie_session'):
        #         ctx.gc3pie_session = self._create_session()
        #     return ctx.gc3pie_session

    @property
    def engine(self):
        return current_app.extensions['gc3pie'].engine
        # ctx = stack.top
        # if ctx is not None:
        #     if not hasattr(ctx, 'gc3pie_engine'):
        #         ctx.gc3pie_engine = self._create_bg_engine()
        #     return ctx.gc3pie_engine

    def get_status_data(self):
        return {
            'stats': self.engine.get_stats_data(),
            'tasks': self.engine.all_tasks_data()
        }

    def get_task_info(self, task):
        return self.engine.get_task_data(task)

    def kill(self, task_id):
        if task_id not in self.session.tasks:
            raise ValueError('No task with id %s within this session.' % task_id)
        else:
            self.engine.kill(self.session.tasks[task_id])
            return True

    # def ready(self):
    #     # try to extract nodename from HTTP request
    #     addr = request.remote_addr
    #     hostname = request.environ['HTTP_REMOTE_HOST'] if ('HTTP_REMOTE_HOST' in request.environ) else None
    #     # look for POST/GET parameters
    #     auth = request.values['auth']
    #     nodename = request.values['hostname']
    #     # perform registration
    #     nodename = nodename.split('.')[0]
    #     log.info("Host '%s' (%s) registering as node '%s'",
    #              (hostname if hostname is not None else "<UNKNOWN>"),
    #              (addr if addr is not None else "unknown address"),
    #              nodename)
    #     self.vm_is_ready(auth, nodename)
    #     return 'OK'