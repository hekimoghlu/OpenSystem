/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#define FUNCTION(f_, t_) { t_ = f_; }
#define MACRO(m_) { typeof(m_) v = m_; }
#define MACRO_VALUE(m_, v_) _Static_assert((m_)==(v_),#m_)
#define MACRO_TYPE(t_, m_) { t_ v = m_; }
#define TYPE(t_) { t_ value; }
#define INCOMPLETE_TYPE(t_) { t_* value; }
#define STRUCT_MEMBER(s_, t_, n_) { s_ s; t_* ptr = &(s.n_); }
#define STRUCT_MEMBER_ARRAY(s_, t_, n_) { s_ s; t_* ptr = &(s.n_[0]); }
#define STRUCT_MEMBER_FUNCTION_POINTER(s_, t_, n_) { s_ s; t_ = (s.n_); }
