/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#ifndef _UAPI_SPI_H
#define _UAPI_SPI_H
#include <linux/const.h>
#define SPI_CPHA _BITUL(0)
#define SPI_CPOL _BITUL(1)
#define SPI_MODE_0 (0 | 0)
#define SPI_MODE_1 (0 | SPI_CPHA)
#define SPI_MODE_2 (SPI_CPOL | 0)
#define SPI_MODE_3 (SPI_CPOL | SPI_CPHA)
#define SPI_MODE_X_MASK (SPI_CPOL | SPI_CPHA)
#define SPI_CS_HIGH _BITUL(2)
#define SPI_LSB_FIRST _BITUL(3)
#define SPI_3WIRE _BITUL(4)
#define SPI_LOOP _BITUL(5)
#define SPI_NO_CS _BITUL(6)
#define SPI_READY _BITUL(7)
#define SPI_TX_DUAL _BITUL(8)
#define SPI_TX_QUAD _BITUL(9)
#define SPI_RX_DUAL _BITUL(10)
#define SPI_RX_QUAD _BITUL(11)
#define SPI_CS_WORD _BITUL(12)
#define SPI_TX_OCTAL _BITUL(13)
#define SPI_RX_OCTAL _BITUL(14)
#define SPI_3WIRE_HIZ _BITUL(15)
#define SPI_RX_CPHA_FLIP _BITUL(16)
#define SPI_MOSI_IDLE_LOW _BITUL(17)
#define SPI_MOSI_IDLE_HIGH _BITUL(18)
#define SPI_MODE_USER_MASK (_BITUL(19) - 1)
#endif
