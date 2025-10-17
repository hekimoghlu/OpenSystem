/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
/*
 * This header provides some gory details to implement the <os/atomic_private.h>
 * interfaces. Nothing in this header should be called directly, no promise is
 * made to keep this interface stable.
 *
 * Architecture overrides.
 */

#ifndef __OS_ATOMIC_PRIVATE_H__
#error "Do not include <os/atomic_private_arch.h> directly, use <os/atomic_private.h>"
#endif

#ifndef __OS_ATOMIC_PRIVATE_ARCH_H__
#define __OS_ATOMIC_PRIVATE_ARCH_H__

#pragma mark - arm v7

#if defined(__arm__)

#if OS_ATOMIC_CONFIG_MEMORY_ORDER_DEPENDENCY
/*
 * On armv7, we do provide fine grained dependency injection, so
 * memory_order_dependency maps to relaxed as far as thread fences are concerned
 */
#undef _os_atomic_mo_dependency
#define _os_atomic_mo_dependency      memory_order_relaxed

#undef os_atomic_make_dependency
#define os_atomic_make_dependency(v) ({ \
	os_atomic_dependency_t _dep; \
	__asm__ __volatile__("and %[_dep], %[_v], #0" \
	    : [_dep] "=r" (_dep.__opaque_zero) \
	    : [_v] "r" (v)); \
	os_compiler_barrier(acquire); \
	_dep; \
})
#endif // OS_ATOMIC_CONFIG_MEMORY_ORDER_DEPENDENCY

#define os_atomic_clear_exclusive()  __builtin_arm_clrex()

#define os_atomic_load_exclusive(p, m)  ({ \
	os_atomic_basetypeof(p) _r = __builtin_arm_ldrex(os_cast_to_nonatomic_pointer(p)); \
	_os_memory_fence_after_atomic(m); \
	_os_compiler_barrier_after_atomic(m); \
	_r; \
})

#define os_atomic_store_exclusive(p, v, m)  ({ \
	_os_compiler_barrier_before_atomic(m); \
	_os_memory_fence_before_atomic(m); \
	!__builtin_arm_strex(v, os_cast_to_nonatomic_pointer(p)); \
})

/*
 * armv7 override of os_atomic_rmw_loop
 * documentation for os_atomic_rmw_loop is in <os/atomic_private.h>
 */
#undef os_atomic_rmw_loop
#define os_atomic_rmw_loop(p, ov, nv, m, ...)  ({ \
	int _result = 0; uint32_t _err = 0; \
	__auto_type *_p = os_cast_to_nonatomic_pointer(p); \
	for (;;) { \
	        ov = __builtin_arm_ldrex(_p); \
	        __VA_ARGS__; \
	        if (!_err) { \
	/* release barrier only done for the first loop iteration */ \
	                _os_memory_fence_before_atomic(m); \
	        } \
	        _err = __builtin_arm_strex(nv, _p); \
	        if (__builtin_expect(!_err, 1)) { \
	                _os_memory_fence_after_atomic(m); \
	                _result = 1; \
	                break; \
	        } \
	} \
	_os_compiler_barrier_after_atomic(m); \
	_result; \
})

/*
 * armv7 override of os_atomic_rmw_loop_give_up
 * documentation for os_atomic_rmw_loop_give_up is in <os/atomic_private.h>
 */
#undef os_atomic_rmw_loop_give_up
#define os_atomic_rmw_loop_give_up(...) \
	({ os_atomic_clear_exclusive(); __VA_ARGS__; break; })

#endif // __arm__

#pragma mark - arm64

#if defined(__arm64__)

#if OS_ATOMIC_CONFIG_MEMORY_ORDER_DEPENDENCY
/*
 * On arm64, we do provide fine grained dependency injection, so
 * memory_order_dependency maps to relaxed as far as thread fences are concerned
 */
#undef _os_atomic_mo_dependency
#define _os_atomic_mo_dependency      memory_order_relaxed

#undef os_atomic_make_dependency
#if __ARM64_ARCH_8_32__
#define os_atomic_make_dependency(v) ({ \
	os_atomic_dependency_t _dep; \
	__asm__ __volatile__("and %w[_dep], %w[_v], wzr" \
	    : [_dep] "=r" (_dep.__opaque_zero) \
	    : [_v] "r" (v)); \
	os_compiler_barrier(acquire); \
	_dep; \
})
#else
#define os_atomic_make_dependency(v) ({ \
	os_atomic_dependency_t _dep; \
	__asm__ __volatile__("and %[_dep], %[_v], xzr" \
	    : [_dep] "=r" (_dep.__opaque_zero) \
	    : [_v] "r" (v)); \
	os_compiler_barrier(acquire); \
	_dep; \
})
#endif
#endif // OS_ATOMIC_CONFIG_MEMORY_ORDER_DEPENDENCY

#if defined(__ARM_ARCH_8_4__)
/* on armv8.4 16-byte aligned load/store pair is atomic */
#undef os_atomic_load_is_plain
#define os_atomic_load_is_plain(p)   (sizeof(*(p)) <= 16)
#endif

#define os_atomic_clear_exclusive()  __builtin_arm_clrex()

#define os_atomic_load_exclusive(p, m)  ({ \
	os_atomic_basetypeof(p) _r = _os_atomic_mo_has_acquire(_os_atomic_mo_##m##_smp) \
	    ? __builtin_arm_ldaex(os_cast_to_nonatomic_pointer(p)) \
	    : __builtin_arm_ldrex(os_cast_to_nonatomic_pointer(p)); \
	_os_compiler_barrier_after_atomic(m); \
	_r; \
})

#define os_atomic_store_exclusive(p, v, m)  ({ \
	_os_compiler_barrier_before_atomic(m); \
	(_os_atomic_mo_has_release(_os_atomic_mo_##m##_smp) \
	    ? !__builtin_arm_stlex(v, os_cast_to_nonatomic_pointer(p)) \
	        : !__builtin_arm_strex(v, os_cast_to_nonatomic_pointer(p))); \
})

#if OS_ATOMIC_USE_LLSC

/*
 * arm64 (without armv81 atomics) override of os_atomic_rmw_loop
 * documentation for os_atomic_rmw_loop is in <os/atomic_private.h>
 */
#undef os_atomic_rmw_loop
#define os_atomic_rmw_loop(p, ov, nv, m, ...)  ({ \
	int _result = 0; \
	__auto_type *_p = os_cast_to_nonatomic_pointer(p); \
	_os_compiler_barrier_before_atomic(m); \
	do { \
	        if (_os_atomic_mo_has_acquire(_os_atomic_mo_##m##_smp)) { \
	                ov = __builtin_arm_ldaex(_p); \
	        } else { \
	                ov = __builtin_arm_ldrex(_p); \
	        } \
	        __VA_ARGS__; \
	        if (_os_atomic_mo_has_release(_os_atomic_mo_##m##_smp)) { \
	                _result = !__builtin_arm_stlex(nv, _p); \
	        } else { \
	                _result = !__builtin_arm_strex(nv, _p); \
	        } \
	} while (__builtin_expect(!_result, 0)); \
	_os_compiler_barrier_after_atomic(m); \
	_result; \
})

/*
 * arm64 override of os_atomic_rmw_loop_give_up
 * documentation for os_atomic_rmw_loop_give_up is in <os/atomic_private.h>
 */
#undef os_atomic_rmw_loop_give_up
#define os_atomic_rmw_loop_give_up(...) \
	({ os_atomic_clear_exclusive(); __VA_ARGS__; break; })

#endif // OS_ATOMIC_USE_LLSC

#endif // __arm64__

#endif /* __OS_ATOMIC_PRIVATE_ARCH_H__ */
