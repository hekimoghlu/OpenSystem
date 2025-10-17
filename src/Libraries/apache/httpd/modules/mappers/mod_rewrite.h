/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
 * @file  mod_rewrite.h
 * @brief Rewrite Extension module for Apache
 *
 * @defgroup MOD_REWRITE mod_rewrite
 * @ingroup APACHE_MODS
 * @{
 */

#ifndef MOD_REWRITE_H
#define MOD_REWRITE_H 1

#include "apr_optional.h"
#include "httpd.h"

#define REWRITE_REDIRECT_HANDLER_NAME "redirect-handler"

/* rewrite map function prototype */
typedef char *(rewrite_mapfunc_t)(request_rec *r, char *key);

/* optional function declaration */
APR_DECLARE_OPTIONAL_FN(void, ap_register_rewrite_mapfunc,
                        (char *name, rewrite_mapfunc_t *func));

#endif /* MOD_REWRITE_H */
/** @} */
