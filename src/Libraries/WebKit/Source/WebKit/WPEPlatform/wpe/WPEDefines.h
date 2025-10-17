/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#ifndef WPEDefines_h
#define WPEDefines_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib.h>

#define WPE_API __attribute__((visibility("default")))

/*
 * The G_DECLARE_DERIVABLE_TYPE macro creates an instance struct that
 * has ParentName as its only member; but to be able to use smart pointers
 * in the private struct it should have also a "priv" member. There is no
 * reasonable way of bending the GLib macro to do what we want; so this is
 * a version of the GLib macro adapted accordingly.
 */
#define WPE_DECLARE_DERIVABLE_TYPE(ModuleObjName, module_obj_name, MODULE, OBJ_NAME, ParentName)        \
    WPE_API GType module_obj_name##_get_type (void);                                                 \
    G_GNUC_BEGIN_IGNORE_DEPRECATIONS                                                                    \
    typedef struct _##ModuleObjName ModuleObjName;                                                      \
    typedef struct _##ModuleObjName##Class ModuleObjName##Class;                                        \
    typedef struct _##ModuleObjName##Private ModuleObjName##Private;                                    \
                                                                                                        \
    _GLIB_DEFINE_AUTOPTR_CHAINUP (ModuleObjName, ParentName)                                            \
    G_DEFINE_AUTOPTR_CLEANUP_FUNC (ModuleObjName##Class, g_type_class_unref)                            \
                                                                                                        \
    G_GNUC_UNUSED static inline ModuleObjName * MODULE##_##OBJ_NAME (gpointer ptr) {                    \
      return G_TYPE_CHECK_INSTANCE_CAST (ptr, module_obj_name##_get_type (), ModuleObjName); }          \
    G_GNUC_UNUSED static inline ModuleObjName##Class * MODULE##_##OBJ_NAME##_CLASS (gpointer ptr) {     \
      return G_TYPE_CHECK_CLASS_CAST (ptr, module_obj_name##_get_type (), ModuleObjName##Class); }      \
    G_GNUC_UNUSED static inline gboolean MODULE##_IS_##OBJ_NAME (gpointer ptr) {                        \
      return G_TYPE_CHECK_INSTANCE_TYPE (ptr, module_obj_name##_get_type ()); }                         \
    G_GNUC_UNUSED static inline gboolean MODULE##_IS_##OBJ_NAME##_CLASS (gpointer ptr) {                \
      return G_TYPE_CHECK_CLASS_TYPE (ptr, module_obj_name##_get_type ()); }                            \
    G_GNUC_UNUSED static inline ModuleObjName##Class * MODULE##_##OBJ_NAME##_GET_CLASS (gpointer ptr) { \
      return G_TYPE_INSTANCE_GET_CLASS (ptr, module_obj_name##_get_type (), ModuleObjName##Class); }    \
    G_GNUC_END_IGNORE_DEPRECATIONS                                                                      \
    struct _##ModuleObjName { ParentName parent_instance; ModuleObjName ## Private *priv; };

#endif /* WPEDefines_h */
