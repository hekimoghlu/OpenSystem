/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#include <sstream>
#include <string>

std::string dumpLuaState(lua_State *L) {
	std::stringstream ostr;
	int i;
	int top = lua_gettop(L);
	ostr << "top=" << top << ":\n";
	for (i = 1; i <= top; ++i) {
		int t = lua_type(L, i);
		switch(t) {
		case LUA_TSTRING:
			ostr << "  " << i << ": '" << lua_tostring(L, i) << "'\n";
			break;
		case LUA_TBOOLEAN:
			ostr << "  " << i << ": " << 
					(lua_toboolean(L, i) ? "true" : "false") << "\n";
			break;
		case LUA_TNUMBER:
			ostr << "  " << i << ": " << lua_tonumber(L, i) << "\n";
			break;
		default:
			ostr << "  " << i << ": TYPE=" << lua_typename(L, t) << "\n";
			break;
		}
	}
	return ostr.str();
}
