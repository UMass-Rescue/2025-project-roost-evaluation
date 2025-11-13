# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
HMA configuration with CLIP extension support.

This configuration extends HMA with CLIP (Contrastive Language–Image Pre-training)
capabilities for semantic image matching.
"""

import logging

from OpenMediaMatch.storage.postgres.impl import DefaultOMMStore
from OpenMediaMatch.utils.fetch_benchmarking import InfiniteRandomExchange
from threatexchange.signal_type.pdq.signal import PdqSignal
from threatexchange.signal_type.md5 import VideoMD5Signal
from threatexchange.content_type.photo import PhotoContent
from threatexchange.content_type.video import VideoContent
from threatexchange.exchanges.impl.static_sample import StaticSampleSignalExchangeAPI
from threatexchange.exchanges.impl.ncmec_api import NCMECSignalExchangeAPI
from threatexchange.exchanges.impl.stop_ncii_api import StopNCIISignalExchangeAPI
from threatexchange.exchanges.impl.fb_threatexchange_api import (
    FBThreatExchangeSignalExchangeAPI,
)

# Import CLIP signal type from the extension
try:
    from tx_extension_clip import CLIPSignal
    CLIP_ENABLED = True
except ImportError:
    CLIP_ENABLED = False
    logging.warning("CLIP extension not available. CLIPSignal will not be enabled.")

# Database configuration
DBUSER = "postgres"
DBPASS = "postgres"
DBHOST = "hma-postgresql"
DBNAME = "media_match"
DATABASE_URI = f"postgresql+psycopg2://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}"

# Role configuration
PRODUCTION = False
ROLE_HASHER = True
ROLE_MATCHER = True
ROLE_CURATOR = True
UI_ENABLED = True

# APScheduler (background threads for development)
TASK_FETCHER = True
TASK_INDEXER = True
TASK_INDEX_CACHE = True

# Core functionality configuration
# Include CLIPSignal if available
signal_types = [PdqSignal, VideoMD5Signal]
if CLIP_ENABLED:
    signal_types.append(CLIPSignal)

STORAGE_IFACE_INSTANCE = DefaultOMMStore(
    signal_types=signal_types,
    content_types=[PhotoContent, VideoContent],
    exchange_types=[
        StaticSampleSignalExchangeAPI,
        InfiniteRandomExchange,  # type: ignore
        FBThreatExchangeSignalExchangeAPI,  # type: ignore
        NCMECSignalExchangeAPI,  # type: ignore
        StopNCIISignalExchangeAPI,
    ],
)

# Debugging stuff
# SQLALCHEMY_ENGINE_LOG_LEVEL = logging.INFO

