/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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

#ifndef _TCLAE_ID
#define _TCLAE_ID

#define TCLAE_NAME			PACKAGE_NAME

#if TARGET_API_MAC_CARBON // das 25/10/00: Carbonization
#define TCLAE_FILENAME			"TclAECarbon"
#else
#define TCLAE_FILENAME			"TclAE"
#endif

#define	TCLAE_MAJOR			2				// BCD (0Ñ99)
#define	TCLAE_MINOR			0				// BCD (0Ñ9)
#define	TCLAE_PATCH			5				// BCD (0Ñ9)
#define	TCLAE_STAGE			finalStage			// {developStage, alphaStage, betaStage, finalStage}
#define TCLAE_PRERELEASE	0				// unsigned binary (0Ñ255)

#define TCLAE_VERSION		PACKAGE_VERSION
#define TCLAE_BASIC_VERSION PACKAGE_VERSION

#endif /* _TCLAE_ID */
