/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#define GMPLS_PSC1   1
#define GMPLS_PSC2   2
#define GMPLS_PSC3   3
#define GMPLS_PSC4   4
#define GMPLS_L2SC  51
#define GMPLS_TSC  100
#define GMPLS_LSC  150
#define GMPLS_FSC  200

extern const struct tok gmpls_link_prot_values[];
extern const struct tok gmpls_switch_cap_values[];
extern const struct tok gmpls_switch_cap_tsc_indication_values[];
extern const struct tok gmpls_encoding_values[];
extern const struct tok gmpls_payload_values[];
extern const struct tok diffserv_te_bc_values[];
extern const struct tok lmp_sd_service_config_cpsa_link_type_values[];
extern const struct tok lmp_sd_service_config_cpsa_signal_type_sdh_values[];
extern const struct tok lmp_sd_service_config_cpsa_signal_type_sonet_values[];
