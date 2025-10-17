/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#pragma once

#if XNU_KERNEL_PRIVATE
//DRKTODO: the decompression side stats should be either made optional or
//per-CPU to avoid cacheline contention

typedef struct {
	uint64_t lz4_compressions;
	uint64_t lz4_compression_failures;
	uint64_t lz4_compressed_bytes;
	uint64_t lz4_wk_compression_delta;
	uint64_t lz4_wk_compression_negative_delta;
	uint64_t lz4_post_wk_compressions;

	uint64_t wk_compressions;
	uint64_t wk_cabstime;
	uint64_t wk_sv_compressions;
	uint64_t wk_mzv_compressions;
	uint64_t wk_compression_failures;
	uint64_t wk_compressed_bytes_total;
	uint64_t wk_compressions_exclusive;
	uint64_t wk_compressed_bytes_exclusive;

	uint64_t wkh_compressions;
	uint64_t wkh_cabstime;
	uint64_t wks_compressions;
	uint64_t wks_cabstime;
	uint64_t wks_compressed_bytes;
	uint64_t wks_compression_failures;
	uint64_t wks_sv_compressions;

	uint64_t lz4_decompressions;
	uint64_t lz4_decompressed_bytes;
	uint64_t uc_decompressions;

	uint64_t wk_decompressions;
	uint64_t wk_dabstime;

	uint64_t wkh_decompressions;
	uint64_t wkh_dabstime;

	uint64_t wks_decompressions;
	uint64_t wks_dabstime;

	uint64_t wk_decompressed_bytes;
	uint64_t wk_sv_decompressions;
} compressor_stats_t;

extern compressor_stats_t compressor_stats;

typedef struct {
	uint32_t lz4_selection_max;
	int32_t wkdm_reeval_threshold;
	int32_t lz4_threshold;
	uint32_t lz4_max_failure_skips;
	uint32_t lz4_max_failure_run_length;
	uint32_t lz4_max_preselects;
	uint32_t lz4_run_preselection_threshold;
	uint32_t lz4_run_continue_bytes;
	uint32_t lz4_profitable_bytes;
} compressor_tuneables_t;

extern compressor_tuneables_t vmctune;

#endif /* XNU_KERNEL_PRIVATE */
