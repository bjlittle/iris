.. include:: ../common_links.inc

|iris_version| |build_date| [unreleased]
****************************************

This document explains the changes made to Iris for this release
(:doc:`View all changes <index>`.)


.. dropdown:: |iris_version| Release Highlights
   :color: primary
   :icon: info
   :animate: fade-in
   :open:

   The highlights for this major/minor release of Iris include:

   * N/A

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!


📢 Announcements
================

#. `@bjlittle`_ migrated the ``SciTools`` social community from ``X`` (formally ``Twitter``)
   to `Bluesky <https://bsky.app/profile/scitools.bsky.social>`__ 🦋. (:pull:`6237`)


✨ Features
===========

#. `@trexfeathers`_ and `@ukmo-ccbunney`_ extended the
   :data:`iris.loading.LOAD_PROBLEMS` capturing to *all* NetCDF objects that are
   added to a :class:`~iris.cube.Cube` during loading, as well as a selection
   of other objects such as :class:`~iris.coord_systems.CoordSystem`. Note this
   includes an improvement to how :class:`~iris.coords.DimCoord` is 'gracefully'
   converted to :class:`~iris.coords.AuxCoord` if it is masked - the mask is
   now preserved when it was not previously. See also: :ref:`load-problems`.
   (:pull:`6465`, :pull:`6529`)

#. `@ESadek-MO`_ made MeshCoords immutable. :class:`iris.MeshCoord`s are now updated automatically when
   changing the attached mesh. All changes to the :class:`iris.MeshCoord` should instead be done to
   the relevant :class:`iris.Coord` located on the attached :class:`iris.MeshXY`. This change also affects 
   the behaviour when calling :attr:`iris.MeshCoord.points` and :attr:`MeshCoord.bounds`, which will return
   real data but will leave the :class:`iris.MeshCoord` (and attached mesh) lazy. (:issue:`4757`, :pull:`6405`)

#. `@pp-mo`_ made it possible for the reference surfaces of derived coordinates, like orography, to be lazy.
   (:pull: 6517).


🐛 Bugs Fixed
=============

#. `@HGWright`_ added a new warning to inform users that the boolean coordinate generated by
   :meth:`iris.coord_categorisation.add_season_membership` is not saveable to netcdf. (:pull:`6305`)

#. `@bouweandela`_ changed the ``convert_units`` method on cubes and coordinates
   so it also converts the values of the attributes ``"actual_range"``,
   ``"valid_max"``, ``"valid_min"``, and ``"valid_range"``. (:pull:`6416`)

#. `@ukmo-ccbunney`_ fixed loading and merging of masked data in scalar ``AuxCoords``.
   (:issue:`3584`, :pull:`6468`)

#. `@stephenworsley`_ fixed the html representation of cubes in Jupyter when coordinates
   share the same name. (:pull:`6476`)

#. `@schlunma`_ fixed loading of netCDF files with coordinates that have
   non-string units. (:issue:`6505`, :pull:`6506`)

#. `@ukmo-ccbunney`_ correctly set the ``bplon`` PP field parameter when saving
   a cube defined on Limited Area Model (LAM) grid to PP format. Activate this
   behaviour with the new Futures flag ``iris.FUTURE.lam_pole_offset=True``.
   (:issue:`3560`, :pull:`6520`)

#. `@stephenworsley`_ fixed incompatibilities with numpy v2.3 affecting arrays of dates and
   array printing. (:pull:`6518`)

#. `@stephenworsley`_ fixed a bug which caused :meth:`~iris.cube.CubeList.concatenate_cube`
   to fail when concatenating over multiple axes. (:pull:`6533`)


💣 Incompatible Changes
=======================

#. N/A

🚀 Performance Enhancements
===========================

#. `@pp-mo`_ implemented automatic rechunking of hybrid (aka factory/derived)
   coordinates to avoid excessive memory usage. (:issue:`6404`, :pull:`6516`)


🔥 Deprecations
===============

#. N/A


🔗 Dependencies
===============

#. N/A


📚 Documentation
================

#. `@trexfeathers`_ and `@ukmo-ccbunney`_ added :ref:`load-problems` to the user
   guide. (:pull:`6529`)

#. `@trexfeathers`_ and `@ukmo-ccbunney`_ added a new user guide page:
   :ref:`iris-philosophy`, for readers who are interested in why Iris is
   designed/maintained the way it is. Initial content: :ref:`code-maintenance`,
   :ref:`load-problems-explanation`, :ref:`filtering-warnings-explanation`.
   (:pull:`6529`)


💼 Internal
===========

#. `@pp-mo`_ replaced the PR-based linkchecks with a daily scheduled link checker based
   on `lychee <https://github.com/lycheeverse/lychee-action>`__.
   (:issue:`4140`, :pull:`6386`)

#. `@trexfeathers`_ added a CI workflow to quickly validate that the
   benchmarking setup is still working. (:pull:`6496`)

#. `@trexfeathers`_ improved the stack trace for errors that occur during
   benchmark data generation, showing developers the root problem at-a-glance
   without needing local replication. (:pull:`6524`)


.. comment
    Whatsnew author names (@github name) in alphabetical order. Note that,
    core dev names are automatically included by the common_links.inc:




.. comment
    Whatsnew resources in alphabetical order:
