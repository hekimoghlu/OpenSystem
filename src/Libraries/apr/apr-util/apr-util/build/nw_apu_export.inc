/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "apu.h"

#undef APU_DECLARE
#undef APU_DECLARE_NONSTD
#undef APU_DECLARE_DATA

/* Preprocess all of the standard APR headers. */
#include "apr_anylock.h"
#include "apr_base64.h"
#include "apr_buckets.h"
#include "apr_crypto.h"
#include "apr_date.h"
#include "apr_dbd.h"
#include "apr_dbm.h"
#include "apr_dbm_private.h"
#include "apr_hooks.h"
#include "apr_ldap.h"
#include "apr_ldap_init.h"
#include "apr_ldap_option.h"
#include "apr_ldap_rebind.h"
#include "apr_ldap_url.h"
#include "apr_md4.h"
#include "apr_md5.h"
#include "apr_memcache.h"
#include "apr_optional.h"
#include "apr_optional_hooks.h"
#include "apr_queue.h"
#include "apr_reslist.h"
#include "apr_rmm.h"
#include "apr_sdbm.h"
#include "apr_sha1.h"
#include "apr_strmatch.h"
#include "apr_thread_pool.h"
#include "apr_uri.h"
#include "apr_uuid.h"
#include "apr_xlate.h"
#include "apr_xml.h"
#include "apu_version.h"
#include "apu_want.h"
