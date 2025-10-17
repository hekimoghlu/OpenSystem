/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
OS_ASSUME_NONNULL_BEGIN
__BEGIN_DECLS

#define os_map_t IN_MAP(,_t)

OS_EXPORT
void
IN_MAP(,_init)(os_map_t *m, os_map_config_t * _Nullable config,
			   int struct_version);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_map_init(os_map_t *m, os_map_config_t * _Nullable config) {
	IN_MAP(,_init)(m, config, OS_MAP_CONFIG_S_VERSION);
}

OS_EXPORT
void
IN_MAP(,_destroy)(os_map_t *m);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_map_destroy(os_map_t *m) {
	IN_MAP(,_destroy)(m);
}

OS_EXPORT
void
IN_MAP(,_insert)(os_map_t *m, os_map_key_t key, void *val);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_map_insert(os_map_t *m, os_map_key_t key, void *val) {
	IN_MAP(,_insert)(m, key, val);
}

OS_EXPORT
void *
IN_MAP(,_find)(os_map_t *m, os_map_key_t key);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void * _Nullable
os_map_find(os_map_t *m, os_map_key_t key) {
	return IN_MAP(,_find)(m, key);
}

OS_EXPORT
void * _Nullable
IN_MAP(,_delete)(os_map_t *m, os_map_key_t key);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void * _Nullable
os_map_delete(os_map_t *m, os_map_key_t key) {
	return IN_MAP(,_delete)(m, key);
}

OS_EXPORT
void
IN_MAP(,_clear)(os_map_t *m,
		OS_NOESCAPE IN_MAP(,_payload_handler_t)_Nullable  handler);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_map_clear(os_map_t *m,
	     OS_NOESCAPE IN_MAP(,_payload_handler_t) _Nullable handler) {
    IN_MAP(,_clear)(m, handler);
}

OS_EXPORT
size_t
IN_MAP(,_count)(os_map_t *m);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline size_t
os_map_count(os_map_t *m) {
	return IN_MAP(,_count)(m);
}

OS_EXPORT
void
IN_MAP(,_foreach)(os_map_t *m,
		  OS_NOESCAPE IN_MAP(,_payload_handler_t) handler);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_map_foreach(os_map_t *m,
	       OS_NOESCAPE IN_MAP(,_payload_handler_t) handler) {
	IN_MAP(,_foreach)(m, handler);
}

__END_DECLS
OS_ASSUME_NONNULL_END
