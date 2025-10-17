/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
 * report_symptoms.c
 * - report symptoms related to address acquisition
 */

/* 
 * Modification History
 *
 * February 23, 2016	Dieter Siegmund (dieter@apple.com)
 * - initial version
 */

#include <stdint.h>
#include <sys/types.h>
#include <SymptomReporter/SymptomReporter.h>
#include "report_symptoms.h"
#include "symbol_scope.h"
#include <dispatch/dispatch.h>

#define SYMPTOM_REPORTER_configd_NUMERIC_ID	0x68
#define SYMPTOM_REPORTER_configd_TEXT_ID	"com.apple.configd"

#define SYMPTOM_ADDRESS_ACQUISITION_FAILED	0x00068001
#define SYMPTOM_ADDRESS_ACQUISITION_SUCCEEDED	0x00068002

#define INTERFACE_INDEX_QUALIFIER		0

PRIVATE_EXTERN bool
report_address_acquisition_symptom(int ifindex, bool success)
{
    STATIC dispatch_once_t 	S_once;
    STATIC symptom_framework_t	S_reporter;
    bool			reported = false;

    dispatch_once(&S_once, ^{
	    S_reporter
		= symptom_framework_init(SYMPTOM_REPORTER_configd_NUMERIC_ID,
					 SYMPTOM_REPORTER_configd_TEXT_ID);
    });

    if (S_reporter != NULL) {
	symptom_ident_t	ident;
	symptom_t 	symptom;

	ident = success ? SYMPTOM_ADDRESS_ACQUISITION_SUCCEEDED
	    : SYMPTOM_ADDRESS_ACQUISITION_FAILED;
	symptom = symptom_new(S_reporter, ident);
	if (symptom != NULL) {
	    symptom_set_qualifier(symptom, (uint64_t)ifindex,
				  INTERFACE_INDEX_QUALIFIER);
	    reported = (symptom_send(symptom) == 0);
	}
    }
    return (reported);
}
