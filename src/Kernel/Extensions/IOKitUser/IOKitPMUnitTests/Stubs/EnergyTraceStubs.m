/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

//
//  EnergyTraceStubs.m
//  IOKitPMUnitTests
//
//  Created by Faramola Isiaka on 7/21/21.
//

#include "EnergyTraceStubs.h"

void entr_act_begin(__unused entr_component_t component, __unused entr_opcode_t activation_opcode, __unused entr_ident_t activation_id, __unused entr_quality_t activation_quality, __unused entr_value_t activation_value)
{
    return;
}

void entr_act_end(__unused entr_component_t component, __unused entr_opcode_t activation_code, __unused entr_ident_t activation_id, __unused entr_quality_t activation_quality, __unused entr_value_t activation_value)
{
    return;
}

void entr_act_modify(__unused entr_component_t component, __unused entr_opcode_t activation_opcode, __unused entr_ident_t activation_id, __unused entr_quality_t modification_quality, __unused entr_value_t modification_value)
{
    return;
}
