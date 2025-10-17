/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <os/variant_private.h>

#include "../libdarwin/variant.c"

#define bool2str(b) (b ? "true" : "false")

static void __dead2
usage(void)
{
	printf("osvariantutil status\n");
	printf("osvariantutil parse <kern.osvariant_status>\n");
	printf("osvariantutil check <OS_VARIANT_STRING\n");
	printf("  For Example: osvariantutil check IsDarwinOS\n");
	printf("  Refer variant_private.h for valid OS_VARIANT_STRING\n");
	exit(1);
}

int
main(int argc, char *argv[]) {
	// Warm up the dispatch_once
	_check_disabled(VP_CONTENT);

	if (argc == 2 && strcmp(argv[1], "status") == 0) {
		uint64_t status = _get_cached_check_status();
		printf("Cached status: %llx\n", status);
	} else if (argc == 3 && strcmp(argv[1], "parse") == 0) {
		uint64_t status = strtoull(argv[2], NULL, 0);
		if ((status & STATUS_INITIAL_BITS) != STATUS_INITIAL_BITS) {
			printf("Invalid status: 0x%llx\n", status);
			exit(1);
		}
		_restore_cached_check_status(status);
		printf("Using status: %llx\n", status);
	} else if (argc == 3 && strcmp(argv[1], "check") == 0) {
          if (os_variant_check("com.apple.osvariantutil", argv[2]) == true) {
            printf("%s: true\n", argv[2]);
            exit(0);
          } else {
            printf("%s: false\n", argv[2]);
            exit(1);
          }
        } else {
		usage();
	}

	printf("\nOS Variants:\n");
	printf("\tos_variant_has_internal_content: %s\n",
			bool2str(os_variant_has_internal_content("com.apple.osvariantutil")));
	printf("\tos_variant_has_internal_diagnostics: %s\n",
			bool2str(os_variant_has_internal_diagnostics("com.apple.osvariantutil")));
	printf("\tos_variant_has_internal_ui: %s\n",
			bool2str(os_variant_has_internal_ui("com.apple.osvariantutil")));
	printf("\tos_variant_allows_internal_security_properties: %s\n",
		   	bool2str(os_variant_allows_internal_security_policies("com.apple.osvariantutil")));
	printf("\tos_variant_has_factory_content: %s\n",
			bool2str(os_variant_has_factory_content("com.apple.osvariantutil")));
	printf("\tos_variant_is_darwinos: %s\n",
			bool2str(os_variant_is_darwinos("com.apple.osvariantutil")));
	printf("\tos_variant_uses_ephemeral_storage: %s\n",
			bool2str(os_variant_uses_ephemeral_storage("com.apple.osvariantutil")));
	printf("\tos_variant_is_recovery: %s\n",
			bool2str(os_variant_is_recovery("com.apple.osvariantutil")));
#if TARGET_OS_OSX
	printf("\tos_variant_is_basesystem: %s\n",
			bool2str(os_variant_is_basesystem("com.apple.osvariantutil")));
#endif
	printf("\tos_variant_has_full_logging: %s\n",
			bool2str(os_variant_check("com.apple.osvariantutil", "HasFullLogging")));

	printf("\nOS Variant Overrides:\n");
	printf("\tCONTENT: %s\n", bool2str(_check_disabled(VP_CONTENT)));
	printf("\tDIAGNOSTICS: %s\n", bool2str(_check_disabled(VP_DIAGNOSTICS)));
	printf("\tUI: %s\n", bool2str(_check_disabled(VP_UI)));
	printf("\tSECURITY: %s\n", bool2str(_check_disabled(VP_SECURITY)));

	printf("\nOS Variant Inputs:\n");
#if !TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
	printf("\tInternal Content: %s\n", bool2str(_check_internal_content()));
#endif
#if TARGET_OS_IPHONE
	printf("\tInternal Release Type: %s\n", bool2str(_check_internal_release_type()));
	printf("\tFactory Release Type: %s\n", bool2str(_check_factory_release_type()));
	printf("\tDarwin Release Type: %s\n", bool2str(_check_darwin_release_type()));
	printf("\tRecovery Release Type: %s\n", bool2str(_check_recovery_release_type()));
	printf("\tDevelopment Kernel: %s\n", bool2str(_check_development_kernel()));
#else
	printf("\tInternal Diags Profile: %s\n", bool2str(_check_internal_diags_profile()));
	printf("\tFactory Content: %s\n", bool2str(_check_factory_content()));
	printf("\tBaseSystem Content: %s\n", bool2str(_check_base_system_content()));
	printf("\tdarwinOS Content: %s\n", bool2str(_check_darwinos_content()));
#endif
	printf("\tCan Has Debugger: %s\n", bool2str(_check_can_has_debugger()));

	return 0;
}
