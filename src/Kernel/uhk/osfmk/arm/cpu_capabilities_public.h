/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef _ARM_CPU_CAPABILITIES_PUBLIC_H
#define _ARM_CPU_CAPABILITIES_PUBLIC_H



/*
 * In order to reduce the number of sysctls require for a process to get
 * the full list of supported processor capabilities extensions, the
 * hw.optional.arm.caps sysctl generates a bit buffer with each bit representing
 * the presence (1) or absence (0) of a given FEAT extension.
 */

#define HW_OPTIONAL_ARM_CAPS

/*
 * Clang needs those bits to remain constant.
 * Existing entries should never be updated as they are ABI.
 * Adding new entries to the end and bumping CAP_BIT_NB is okay.
 */

#define CAP_BIT_FEAT_FlagM          0
#define CAP_BIT_FEAT_FlagM2         1
#define CAP_BIT_FEAT_FHM            2
#define CAP_BIT_FEAT_DotProd        3
#define CAP_BIT_FEAT_SHA3           4
#define CAP_BIT_FEAT_RDM            5
#define CAP_BIT_FEAT_LSE            6
#define CAP_BIT_FEAT_SHA256         7
#define CAP_BIT_FEAT_SHA512         8
#define CAP_BIT_FEAT_SHA1           9
#define CAP_BIT_FEAT_AES            10
#define CAP_BIT_FEAT_PMULL          11
#define CAP_BIT_FEAT_SPECRES        12
#define CAP_BIT_FEAT_SB             13
#define CAP_BIT_FEAT_FRINTTS        14
#define CAP_BIT_FEAT_LRCPC          15
#define CAP_BIT_FEAT_LRCPC2         16
#define CAP_BIT_FEAT_FCMA           17
#define CAP_BIT_FEAT_JSCVT          18
#define CAP_BIT_FEAT_PAuth          19
#define CAP_BIT_FEAT_PAuth2         20
#define CAP_BIT_FEAT_FPAC           21
#define CAP_BIT_FEAT_DPB            22
#define CAP_BIT_FEAT_DPB2           23
#define CAP_BIT_FEAT_BF16           24
#define CAP_BIT_FEAT_I8MM           25
#define CAP_BIT_FEAT_WFxT           26
#define CAP_BIT_FEAT_RPRES          27
#define CAP_BIT_FEAT_ECV            28
#define CAP_BIT_FEAT_AFP            29
#define CAP_BIT_FEAT_LSE2           30
#define CAP_BIT_FEAT_CSV2           31
#define CAP_BIT_FEAT_CSV3           32
#define CAP_BIT_FEAT_DIT            33
#define CAP_BIT_FEAT_FP16           34
#define CAP_BIT_FEAT_SSBS           35
#define CAP_BIT_FEAT_BTI            36


/* SME */
#define CAP_BIT_FEAT_SME            40
#define CAP_BIT_FEAT_SME2           41
#define CAP_BIT_FEAT_SME_F64F64     42
#define CAP_BIT_FEAT_SME_I16I64     43

#define CAP_BIT_AdvSIMD             49
#define CAP_BIT_AdvSIMD_HPFPCvt     50
#define CAP_BIT_FEAT_CRC32          51

#define CAP_BIT_SME_F32F32          52
#define CAP_BIT_SME_BI32I32         53
#define CAP_BIT_SME_B16F32          54
#define CAP_BIT_SME_F16F32          55
#define CAP_BIT_SME_I8I32           56
#define CAP_BIT_SME_I16I32          57

#define CAP_BIT_FEAT_PACIMP         58


#define CAP_BIT_FEAT_HBC            64
#define CAP_BIT_FEAT_EBF16          65
#define CAP_BIT_FEAT_SPECRES2       66
#define CAP_BIT_FEAT_CSSC           67
#define CAP_BIT_FEAT_FPACCOMBINE    68


#define CAP_BIT_FP_SyncExceptions   73

/* Legacy definitions for backwards compatibility */
#define CAP_BIT_CRC32               CAP_BIT_FEAT_CRC32

/* Total number of FEAT bits. */
#define CAP_BIT_NB 74

#endif /* _ARM_CPU_CAPABILITIES_PUBLIC_H */
