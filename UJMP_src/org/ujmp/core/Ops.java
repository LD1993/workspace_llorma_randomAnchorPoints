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

package org.ujmp.core;

import org.ujmp.core.calculation.DivideMatrix;
import org.ujmp.core.calculation.DivideMatrixCalculation;
import org.ujmp.core.calculation.DivideScalar;
import org.ujmp.core.calculation.DivideScalarCalculation;
import org.ujmp.core.calculation.MinusMatrix;
import org.ujmp.core.calculation.MinusMatrixCalculation;
import org.ujmp.core.calculation.MinusScalar;
import org.ujmp.core.calculation.MinusScalarCalculation;
import org.ujmp.core.calculation.Mtimes;
import org.ujmp.core.calculation.MtimesCalculation;
import org.ujmp.core.calculation.PlusMatrix;
import org.ujmp.core.calculation.PlusMatrixCalculation;
import org.ujmp.core.calculation.PlusScalar;
import org.ujmp.core.calculation.PlusScalarCalculation;
import org.ujmp.core.calculation.TimesMatrix;
import org.ujmp.core.calculation.TimesMatrixCalculation;
import org.ujmp.core.calculation.TimesScalar;
import org.ujmp.core.calculation.TimesScalarCalculation;
import org.ujmp.core.calculation.Transpose;
import org.ujmp.core.calculation.TransposeCalculation;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Chol;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Eig;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Inv;
import org.ujmp.core.doublematrix.calculation.general.decomposition.LU;
import org.ujmp.core.doublematrix.calculation.general.decomposition.QR;
import org.ujmp.core.doublematrix.calculation.general.decomposition.SVD;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Solve;
import org.ujmp.core.util.AbstractPlugin;

/**
 * @deprecated use <code>Matrix.[operation]</code> instead
 */
public abstract class Ops {

	public static TransposeCalculation<Matrix, Matrix> transpose = Transpose.MATRIX;

	public static PlusMatrixCalculation<Matrix, Matrix, Matrix> plusMatrix = PlusMatrix.MATRIX;

	public static MinusMatrixCalculation<Matrix, Matrix, Matrix> minusMatrix = MinusMatrix.MATRIX;

	public static TimesMatrixCalculation<Matrix, Matrix, Matrix> timesMatrix = TimesMatrix.MATRIX;

	public static DivideMatrixCalculation<Matrix, Matrix, Matrix> divideMatrix = DivideMatrix.MATRIX;

	public static PlusScalarCalculation<Matrix, Matrix> plusScalar = PlusScalar.MATRIX;

	public static MinusScalarCalculation<Matrix, Matrix> minusScalar = MinusScalar.MATRIX;

	public static TimesScalarCalculation<Matrix, Matrix> timesScalar = TimesScalar.MATRIX;

	public static DivideScalarCalculation<Matrix, Matrix> divideScalar = DivideScalar.MATRIX;

	public static MtimesCalculation<Matrix, Matrix, Matrix> mtimes = Mtimes.MATRIX;

	public static SVD<Matrix> svd = org.ujmp.core.doublematrix.calculation.general.decomposition.SVD.INSTANCE;

	public static LU<Matrix> lu = org.ujmp.core.doublematrix.calculation.general.decomposition.LU.INSTANCE;

	public static QR<Matrix> qr = org.ujmp.core.doublematrix.calculation.general.decomposition.QR.INSTANCE;

	public static Inv<Matrix> inv = Inv.INSTANCE;

	public static Solve<Matrix> solve = Solve.INSTANCE;

	public static Chol<Matrix> chol = Chol.INSTANCE;

	public static Eig<Matrix> eig = Eig.INSTANCE;

	
}
