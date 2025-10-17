/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

//------------------------------------------------------------------------------
/*
  https://github.com/vinniefalco/LuaBridge
  
  Copyright 2012, Vinnie Falco <vinnie.falco@gmail.com>
  Copyright 2008, Nigel Atkinson <suprapilot+LuaCode@gmail.com>

  License: The MIT License (http://www.opensource.org/licenses/mit-license.php)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
//==============================================================================

class LuaException : public std::exception 
{
private:
  lua_State* m_L;
  std::string m_what;

public:
  //----------------------------------------------------------------------------
  /**
      Construct a LuaException after a lua_pcall().
  */
  LuaException (lua_State* L, int /*code*/)
    : m_L (L)
  {
    whatFromStack ();
  }

  //----------------------------------------------------------------------------

  LuaException (lua_State *L,
                char const*,
                char const*,
                long)
    : m_L (L)
  {
    whatFromStack ();
  }

  //----------------------------------------------------------------------------

  ~LuaException() throw ()
  {
  }

  //----------------------------------------------------------------------------

  char const* what() const throw ()
  {
    return m_what.c_str();
  }

  //============================================================================
  /**
      Throw an exception.

      This centralizes all the exceptions thrown, so that we can set
      breakpoints before the stack is unwound, or otherwise customize the
      behavior.
  */
  template <class Exception>
  static void Throw (Exception e)
  {
    throw e;
  }

  //----------------------------------------------------------------------------
  /**
      Wrapper for lua_pcall that throws.
  */
  static void pcall (lua_State* L, int nargs = 0, int nresults = 0, int msgh = 0)
  {
    int code = lua_pcall (L, nargs, nresults, msgh);

    if (code != LUABRIDGE_LUA_OK)
      Throw (LuaException (L, code));
  }

  //----------------------------------------------------------------------------

protected:
  void whatFromStack ()
  {
    if (lua_gettop (m_L) > 0)
    {
      char const* s = lua_tostring (m_L, -1);
      m_what = s ? s : "";
    }
    else
    {
      // stack is empty
      m_what = "missing error";
    }
  }
};
