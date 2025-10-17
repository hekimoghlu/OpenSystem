/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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


#include "zfstream.h"

int main() {

  // Construct a stream object with this filebuffer.  Anything sent
  // to this stream will go to standard out.
  gzofstream os( 1, ios::out );

  // This text is getting compressed and sent to stdout.
  // To prove this, run 'test | zcat'.
  os << "Hello, Mommy" << endl;

  os << setcompressionlevel( Z_NO_COMPRESSION );
  os << "hello, hello, hi, ho!" << endl;

  setcompressionlevel( os, Z_DEFAULT_COMPRESSION )
    << "I'm compressing again" << endl;

  os.close();

  return 0;

}
