/*
 * Copyright (C) 2008-2010 by Holger Arndt
 *
 * This file is part of the Universal Java Matrix Package (UJMP).
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * UJMP is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * UJMP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with UJMP; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package org.ujmp.lsimpute;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.ujmp.core.util.AbstractPlugin;

public class Plugin extends AbstractPlugin {

	private final List<Object> dependencies = new ArrayList<Object>();

	private final List<String> neededClasses = new ArrayList<String>();

	public Plugin() {
		dependencies.add("ujmp-core");
		dependencies.add("LSimpute.jar");
		neededClasses.add("edu.uci.ics.jung.graph.Graph");
		neededClasses.add("findp");
	}

	
	public String getDescription() {
		return "library for imputing missing values using the least sqaures method";
	}

	
	public Collection<Object> getDependencies() {
		return dependencies;
	}

	
	public Collection<String> getNeededClasses() {
		return neededClasses;
	}

}
