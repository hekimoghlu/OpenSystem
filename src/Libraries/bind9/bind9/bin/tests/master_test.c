/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
/* $Id: master_test.c,v 1.32 2009/09/02 23:48:01 tbox Exp $ */

#include <config.h>

#include <stdlib.h>
#include <string.h>

#include <isc/buffer.h>
#include <isc/mem.h>
#include <isc/print.h>
#include <isc/util.h>

#include <dns/callbacks.h>
#include <dns/master.h>
#include <dns/name.h>
#include <dns/rdataset.h>
#include <dns/result.h>

isc_mem_t *mctx;

static isc_result_t
print_dataset(void *arg, dns_name_t *owner, dns_rdataset_t *dataset) {
	char buf[64*1024];
	isc_buffer_t target;
	isc_result_t result;

	UNUSED(arg);

	isc_buffer_init(&target, buf, 64*1024);
	result = dns_rdataset_totext(dataset, owner, ISC_FALSE, ISC_FALSE,
				     &target);
	if (result == ISC_R_SUCCESS)
		fprintf(stdout, "%.*s\n", (int)target.used,
					  (char*)target.base);
	else
		fprintf(stdout, "dns_rdataset_totext: %s\n",
			dns_result_totext(result));

	return (ISC_R_SUCCESS);
}

int
main(int argc, char *argv[]) {
	isc_result_t result;
	dns_name_t origin;
	isc_buffer_t source;
	isc_buffer_t target;
	unsigned char name_buf[255];
	dns_rdatacallbacks_t callbacks;

	UNUSED(argc);

	RUNTIME_CHECK(isc_mem_create(0, 0, &mctx) == ISC_R_SUCCESS);

	if (argv[1]) {
		isc_buffer_init(&source, argv[1], strlen(argv[1]));
		isc_buffer_add(&source, strlen(argv[1]));
		isc_buffer_setactive(&source, strlen(argv[1]));
		isc_buffer_init(&target, name_buf, 255);
		dns_name_init(&origin, NULL);
		result = dns_name_fromtext(&origin, &source, dns_rootname,
					   0, &target);
		if (result != ISC_R_SUCCESS) {
			fprintf(stdout, "dns_name_fromtext: %s\n",
				dns_result_totext(result));
			exit(1);
		}

		dns_rdatacallbacks_init_stdio(&callbacks);
		callbacks.add = print_dataset;

		result = dns_master_loadfile(argv[1], &origin, &origin,
					     dns_rdataclass_in, 0,
					     &callbacks, mctx);
		fprintf(stdout, "dns_master_loadfile: %s\n",
			dns_result_totext(result));
	}
	return (0);
}
