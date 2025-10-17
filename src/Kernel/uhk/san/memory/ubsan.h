/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#ifndef _UBSAN_H_
#define _UBSAN_H_

#include <stdint.h>
#include <stdbool.h>

struct san_type_desc {
	uint16_t type; // 0: integer, 1: float
	union {
		struct {
			uint16_t issigned : 1;
			uint16_t width    : 15;
		}; /* int descriptor */
		struct {
			uint16_t float_desc;
		}; /* float descriptor */
	};
	const char name[];
};

struct san_src_loc {
	const char *filename;
	uint32_t line;
	uint32_t col;
};

struct ubsan_overflow_desc {
	struct san_src_loc loc;
	struct san_type_desc *ty;
};

struct ubsan_unreachable_desc {
	struct san_src_loc loc;
};

struct ubsan_shift_desc {
	struct san_src_loc loc;
	struct san_type_desc *lhs_t;
	struct san_type_desc *rhs_t;
};

struct ubsan_align_desc {
	struct san_src_loc loc;
	struct san_type_desc *ty;
	uint8_t align;
	uint8_t kind;
};

struct ubsan_ptroverflow_desc {
	struct san_src_loc loc;
};

struct ubsan_oob_desc {
	struct san_src_loc loc;
	struct san_type_desc *array_ty;
	struct san_type_desc *index_ty;
};

struct ubsan_load_invalid_desc {
	struct san_src_loc loc;
	struct san_type_desc *type;
};

struct ubsan_nullability_arg_desc {
	struct san_src_loc loc;
	struct san_src_loc attr_loc;
	int arg_index;
};

struct ubsan_nullability_ret_desc {
	struct san_src_loc loc;
};

struct ubsan_missing_ret_desc {
	struct san_src_loc loc;
};

struct ubsan_float_desc {
	struct san_src_loc loc;
	struct san_type_desc *type_from;
	struct san_type_desc *type_to;
};

struct ubsan_implicit_conv_desc {
	struct san_src_loc loc;
	struct san_type_desc *type_from;
	struct san_type_desc *type_to;
	unsigned char kind;
};

struct ubsan_func_type_mismatch_desc {
	struct san_src_loc loc;
	struct san_type_desc *type;
};

struct ubsan_vla_bound_desc {
	struct san_src_loc loc;
	struct san_type_desc *type;
};

struct ubsan_invalid_builtin {
	struct san_src_loc loc;
	unsigned char kind;
};

OS_ENUM(ubsan_violation_type, uint8_t,
    UBSAN_OVERFLOW_add = 1,
    UBSAN_OVERFLOW_sub,
    UBSAN_OVERFLOW_mul,
    UBSAN_OVERFLOW_divrem,
    UBSAN_OVERFLOW_negate,
    UBSAN_UNREACHABLE,
    UBSAN_SHIFT,
    UBSAN_ALIGN,
    UBSAN_POINTER_OVERFLOW,
    UBSAN_OOB,
    UBSAN_TYPE_MISMATCH,
    UBSAN_LOAD_INVALID_VALUE,
    UBSAN_NULLABILITY_ARG,
    UBSAN_NULLABILITY_RETURN,
    UBSAN_MISSING_RETURN,
    UBSAN_FLOAT_CAST_OVERFLOW,
    UBSAN_IMPLICIT_CONVERSION,
    UBSAN_FUNCTION_TYPE_MISMATCH,
    UBSAN_VLA_BOUND_NOT_POSITIVE,
    UBSAN_INVALID_BUILTIN,
    UBSAN_VIOLATION_MAX
    );

typedef struct ubsan_violation {
	ubsan_violation_type_t ubsan_type;
	uint64_t lhs;
	uint64_t rhs;
	union {
		struct ubsan_overflow_desc *overflow;
		struct ubsan_unreachable_desc *unreachable;
		struct ubsan_shift_desc *shift;
		struct ubsan_align_desc *align;
		struct ubsan_ptroverflow_desc *ptroverflow;
		struct ubsan_oob_desc *oob;
		struct ubsan_load_invalid_desc *invalid;
		struct ubsan_nullability_arg_desc *nonnull_arg;
		struct ubsan_nullability_ret_desc *nonnull_ret;
		struct ubsan_missing_ret_desc *missing_ret;
		struct ubsan_float_desc *flt;
		struct ubsan_implicit_conv_desc *implicit;
		struct ubsan_func_type_mismatch_desc *func_mismatch;
		struct ubsan_vla_bound_desc *vla_bound;
		struct ubsan_invalid_builtin *invalid_builtin;
		const char *func;
	};
	struct san_src_loc *loc;
} ubsan_violation_t;

typedef struct ubsan_buf {
	char    *ub_buf;
	size_t  ub_buf_size;
	size_t  ub_written;
	bool    ub_err;
} ubsan_buf_t;

void    ubsan_log_append(ubsan_violation_t *);

void    ubsan_json_init(ubsan_buf_t *, char *, size_t);
void    ubsan_json_begin(ubsan_buf_t *, size_t);
size_t  ubsan_json_finish(ubsan_buf_t *);
bool    ubsan_json_format(ubsan_violation_t *, ubsan_buf_t *);

/*
 * UBSan ABI
 */
void __ubsan_handle_add_overflow(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_add_overflow_abort(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_builtin_unreachable(struct ubsan_unreachable_desc *);
void __ubsan_handle_divrem_overflow(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_divrem_overflow_abort(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_float_cast_overflow(struct ubsan_float_desc *, uint64_t);
void __ubsan_handle_float_cast_overflow_abort(struct ubsan_float_desc *, uint64_t);
void __ubsan_handle_function_type_mismatch(struct ubsan_func_type_mismatch_desc*, uint64_t);
void __ubsan_handle_function_type_mismatch_abort(struct ubsan_func_type_mismatch_desc *, uint64_t);
void __ubsan_handle_implicit_conversion(struct ubsan_implicit_conv_desc *, uint64_t, uint64_t);
void __ubsan_handle_implicit_conversion_abort(struct ubsan_implicit_conv_desc *, uint64_t, uint64_t);
void __ubsan_handle_invalid_builtin(struct ubsan_invalid_builtin *);
void __ubsan_handle_invalid_builtin_abort(struct ubsan_invalid_builtin *);
void __ubsan_handle_load_invalid_value(struct ubsan_load_invalid_desc *, uint64_t);
void __ubsan_handle_load_invalid_value_abort(struct ubsan_load_invalid_desc *, uint64_t);
void __ubsan_handle_missing_return(struct ubsan_missing_ret_desc *);
void __ubsan_handle_missing_return_abort(struct ubsan_missing_ret_desc *);
void __ubsan_handle_mul_overflow(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_mul_overflow_abort(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_negate_overflow(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_negate_overflow_abort(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_nonnull_arg(struct ubsan_nullability_arg_desc *);
void __ubsan_handle_nonnull_arg_abort(struct ubsan_nullability_arg_desc *);
void __ubsan_handle_nonnull_return_v1(struct ubsan_nullability_ret_desc *, uint64_t);
void __ubsan_handle_nonnull_return_v1_abort(struct ubsan_nullability_ret_desc *, uint64_t);
void __ubsan_handle_nullability_arg(struct ubsan_nullability_arg_desc *);
void __ubsan_handle_nullability_arg_abort(struct ubsan_nullability_arg_desc *);
void __ubsan_handle_nullability_return_v1(struct ubsan_nullability_ret_desc *, uint64_t);
void __ubsan_handle_nullability_return_v1_abort(struct ubsan_nullability_ret_desc *, uint64_t);
void __ubsan_handle_out_of_bounds(struct ubsan_oob_desc *, uint64_t idx);
void __ubsan_handle_out_of_bounds_abort(struct ubsan_oob_desc *, uint64_t idx);
void __ubsan_handle_pointer_overflow(struct ubsan_ptroverflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_pointer_overflow_abort(struct ubsan_ptroverflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_shift_out_of_bounds(struct ubsan_shift_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_shift_out_of_bounds_abort(struct ubsan_shift_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_sub_overflow(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_sub_overflow_abort(struct ubsan_overflow_desc *, uint64_t lhs, uint64_t rhs);
void __ubsan_handle_type_mismatch_v1(struct ubsan_align_desc *, uint64_t val);
void __ubsan_handle_type_mismatch_v1_abort(struct ubsan_align_desc *, uint64_t val);
void __ubsan_handle_vla_bound_not_positive(struct ubsan_vla_bound_desc *, uint64_t);
void __ubsan_handle_vla_bound_not_positive_abort(struct ubsan_vla_bound_desc *, uint64_t);

#endif /* _UBSAN_H_ */
