/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
// Do nothing if this header is used on it's own
#ifdef IN_SET

OS_ASSUME_NONNULL_BEGIN
__BEGIN_DECLS

#define os_set_t IN_SET(,_t)

OS_EXPORT
void
IN_SET(,_init)(os_set_t *s, os_set_config_t * _Nullable config,
			   int struct_version);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_set_init(os_set_t *s, os_set_config_t * _Nullable config) {
	IN_SET(,_init)(s, config, OS_SET_CONFIG_S_VERSION);
}

OS_EXPORT
void
IN_SET(,_destroy)(os_set_t *s);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_set_destroy(os_set_t *s) {
	IN_SET(,_destroy)(s);
}

OS_EXPORT
void
IN_SET(,_insert)(os_set_t *s, os_set_insert_val_t val);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_set_insert(os_set_t *s, os_set_insert_val_t val) {
	IN_SET(,_insert)(s, val);
}

OS_EXPORT
void *
IN_SET(,_find)(os_set_t *s, os_set_find_val_t val);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void * _Nullable
os_set_find(os_set_t *s, os_set_find_val_t val) {
	return IN_SET(,_find)(s, val);
}

OS_EXPORT
void *
IN_SET(,_delete)(os_set_t *s, os_set_find_val_t val);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void * _Nullable
os_set_delete(os_set_t *s, os_set_find_val_t val) {
	return IN_SET(,_delete)(s, val);
}

OS_EXPORT
void
IN_SET(,_clear)(os_set_t *s, OS_NOESCAPE IN_SET(,_payload_handler_t) handler);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_set_clear(os_set_t *s, OS_NOESCAPE IN_SET(,_payload_handler_t) _Nullable handler) {
	IN_SET(,_clear)(s, handler);
}

OS_EXPORT
size_t
IN_SET(,_count)(os_set_t *s);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline size_t
os_set_count(os_set_t *s) {
	return IN_SET(,_count)(s);
}

OS_EXPORT
void
IN_SET(,_foreach)(os_set_t *s, OS_NOESCAPE IN_SET(,_payload_handler_t) handler);

OS_OVERLOADABLE OS_ALWAYS_INLINE
static inline void
os_set_foreach(os_set_t *s, OS_NOESCAPE IN_SET(,_payload_handler_t) handler) {
	IN_SET(,_foreach)(s, handler);
}

#undef os_set_t

__END_DECLS
OS_ASSUME_NONNULL_END

#endif // IN_SET
