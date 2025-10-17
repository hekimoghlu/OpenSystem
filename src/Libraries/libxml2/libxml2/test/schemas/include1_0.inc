/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

<?xml version="1.0"?>
<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" >

	<xsd:simpleType name="barType">
		<xsd:restriction base="xsd:string">
			<xsd:maxLength value="10" />
		</xsd:restriction>
	</xsd:simpleType>

	<xsd:attribute name="bar" type="barType" fixed="o"/>

	<xsd:element name="bar.B" type="xsd:anyURI" block="#all" nillable="true" final="restriction extension"/>
	
	<xsd:element name="bar">
		<xsd:complexType>
			<xsd:sequence>
				<xsd:element name="bar.A" block="restriction" form="qualified" nillable="true" />
				<xsd:element ref="bar.B"/>
			</xsd:sequence>
			
			<xsd:attribute ref="bar" xml:lang="DE">
				<xsd:annotation/>
			</xsd:attribute>
			<xsd:attribute name="uri" type="xsd:token" />
		</xsd:complexType>
	</xsd:element>

</xsd:schema>