/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#ifndef _EAP8021X_SIM_SIMULATOR_H
#define _EAP8021X_SIM_SIMULATOR_H

#define SIM_KI_SIZE		16
#define SIM_OPC_SIZE	16

void
sim_simulator_gsm_milenage_algo(const uint8_t *opc, const uint8_t *Ki, const uint8_t *rand, uint8_t *sres, uint8_t *Kc);

#endif /* _EAP8021X_SIM_SIMULATOR_H */
