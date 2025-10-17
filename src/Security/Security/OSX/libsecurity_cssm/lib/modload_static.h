/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
// modload_static - pseudo-loading of statically linked plugins
//
#ifndef _H_MODLOAD_STATIC
#define _H_MODLOAD_STATIC

#include "modloader.h"
#include "cssmint.h"
#include <Security/cssmspi.h>
#include <security_cdsa_utilities/callback.h>


namespace Security {


//
// A "plugin" implementation that uses statically linked entry points
//
class StaticPlugin : public Plugin {
public: 
	StaticPlugin(const PluginFunctions &funcs) : entries(funcs) { }

    void load()				{ }
    void unload()			{ }
    bool isLoaded() const	{ return true; }

	CSSM_SPI_ModuleLoadFunction load;
	CSSM_SPI_ModuleUnloadFunction unload;
	CSSM_SPI_ModuleAttachFunction attach;
	CSSM_SPI_ModuleDetachFunction detach;
    
private:
	const PluginFunctions &entries;
};


} // end namespace Security


#endif //_H_MODLOAD_STATIC
