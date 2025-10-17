/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
/**
 * @file  mod_request.h
 * @brief mod_request private header file
 *
 * @defgroup MOD_REQUEST mod_request
 * @ingroup  APACHE_MODS
 * @{
 */

#ifndef MOD_REQUEST_H
#define MOD_REQUEST_H

#include "apr.h"
#include "apr_buckets.h"
#include "apr_optional.h"

#include "httpd.h"
#include "util_filter.h"


#ifdef __cplusplus
extern "C" {
#endif

extern module AP_MODULE_DECLARE_DATA request_module;

#define KEEP_BODY_FILTER "KEEP_BODY"
#define KEPT_BODY_FILTER "KEPT_BODY"

/**
 * Core per-directory configuration.
 */
typedef struct {
    apr_off_t keep_body;
    int keep_body_set;
} request_dir_conf;

APR_DECLARE_OPTIONAL_FN(void, ap_request_insert_filter, (request_rec * r));

APR_DECLARE_OPTIONAL_FN(void, ap_request_remove_filter, (request_rec * r));

#ifdef __cplusplus
}
#endif

#endif /* !MOD_REQUEST_H */
/** @} */
