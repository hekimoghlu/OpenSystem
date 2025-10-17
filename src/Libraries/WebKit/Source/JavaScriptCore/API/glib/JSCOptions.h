/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#if !defined(__JSC_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <jsc/jsc.h> can be included directly."
#endif

#ifndef JSCOptions_h
#define JSCOptions_h

#include <glib-object.h>
#include <jsc/JSCDefines.h>

G_BEGIN_DECLS

#define JSC_OPTIONS_USE_JIT   "useJIT"
#define JSC_OPTIONS_USE_DFG   "useDFGJIT"
#define JSC_OPTIONS_USE_FTL   "useFTLJIT"
#define JSC_OPTIONS_USE_LLINT "useLLInt"

JSC_API gboolean
jsc_options_set_boolean       (const char *option,
                               gboolean    value);
JSC_API gboolean
jsc_options_get_boolean       (const char *option,
                               gboolean   *value);

JSC_API gboolean
jsc_options_set_int           (const char *option,
                               gint        value);
JSC_API gboolean
jsc_options_get_int           (const char *option,
                               gint       *value);

JSC_API gboolean
jsc_options_set_uint          (const char *option,
                               guint       value);
JSC_API gboolean
jsc_options_get_uint          (const char *option,
                               guint      *value);

JSC_API gboolean
jsc_options_set_size          (const char *option,
                               gsize       value);
JSC_API gboolean
jsc_options_get_size          (const char *option,
                               gsize      *value);

JSC_API gboolean
jsc_options_set_double        (const char *option,
                               gdouble     value);
JSC_API gboolean
jsc_options_get_double        (const char *option,
                               gdouble    *value);

JSC_API gboolean
jsc_options_set_string        (const char *option,
                               const char *value);
JSC_API gboolean
jsc_options_get_string        (const char *option,
                               char       **value);

JSC_API gboolean
jsc_options_set_range_string  (const char *option,
                               const char *value);
JSC_API gboolean
jsc_options_get_range_string  (const char *option,
                               char       **value);

typedef enum {
    JSC_OPTION_BOOLEAN,
    JSC_OPTION_INT,
    JSC_OPTION_UINT,
    JSC_OPTION_SIZE,
    JSC_OPTION_DOUBLE,
    JSC_OPTION_STRING,
    JSC_OPTION_RANGE_STRING
} JSCOptionType;

typedef gboolean (* JSCOptionsFunc) (const char    *option,
                                     JSCOptionType  type,
                                     const char    *description,
                                     gpointer       user_data);

JSC_API void
jsc_options_foreach                 (JSCOptionsFunc function,
                                     gpointer       user_data);

JSC_API GOptionGroup *
jsc_options_get_option_group        (void);

G_END_DECLS

#endif /* JSCOptions_h */
