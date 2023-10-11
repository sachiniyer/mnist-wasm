use ndarray::{stack, Array1, Array2, ArrayView1, Axis};

pub struct ActivationFunctions;

impl ActivationFunctions {
    pub fn relu1d(x: Array1<f64>) -> Array1<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    pub fn relu2d(x: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    pub fn relu_backward1d(x: Array1<f64>, y: Array1<f64>) -> Array1<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * y
    }

    pub fn relu_backward2d(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * y
    }

    pub fn logsoftmax1d(x: Array1<f64>) -> Array1<f64> {
        let max_x = x.fold(f64::NAN, |a, b| a.max(*b));
        let diff_x = x.mapv(|x| x - max_x);
        let sum_x = diff_x.mapv(f64::exp).sum().ln();
        diff_x.mapv(|x| x - sum_x)
    }

    pub fn logsoftmax2d(x: Array2<f64>) -> Array2<f64> {
        let max_x = x.fold_axis(ndarray::Axis(1), f64::NAN, |&a, &b| a.max(b));
        let diff_x = &x - max_x.insert_axis(ndarray::Axis(1));
        let sum_x = diff_x.mapv(f64::exp).sum_axis(ndarray::Axis(1));
        let log_sum_x = sum_x.mapv(f64::ln);
        diff_x - log_sum_x.insert_axis(ndarray::Axis(1))
    }

    pub fn logsoftmax_backward1d(x: Array1<f64>, y: Array1<f64>) -> Array1<f64> {
        let softmax_x = (&x - x.fold(f64::NAN, |a, b| a.max(*b))).mapv(f64::exp);
        let softmax_sum = softmax_x.sum();
        let softmax = softmax_x / softmax_sum;
        let n = x.len();
        let delta_ij = Array2::eye(n);
        let softmax_matrix = softmax.broadcast(n).unwrap().to_owned();
        let derivative = &delta_ij - &softmax_matrix;
        y.dot(&derivative)
    }

    pub fn logsoftmax_backward2d(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
        let softmax_x = (&x
            - &x.fold_axis(ndarray::Axis(1), f64::NAN, |&a, &b| a.max(b))
                .insert_axis(ndarray::Axis(1)))
            .mapv(f64::exp);
        let softmax_sum = softmax_x
            .sum_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));
        let softmax = softmax_x / &softmax_sum;
        let n = x.shape()[1];
        let m = x.shape()[0];
        let inner_delta_ij: Array2<f64> = Array2::eye(n);
        let delta_ij = inner_delta_ij.broadcast((m, n, n)).unwrap().to_owned();
        let softmax_matrix = softmax
            .insert_axis(Axis(1))
            .broadcast((m, n, n))
            .unwrap()
            .to_owned();
        let derivative = &delta_ij - &softmax_matrix;
        stack(
            Axis(0),
            &y.axis_iter(Axis(0))
                .zip(derivative.axis_iter(Axis(0)))
                .map(|(y, derivative)| y.dot(&derivative))
                .collect::<Vec<Array1<f64>>>()
                .iter()
                .map(|x| x.view())
                .collect::<Vec<ArrayView1<f64>>>(),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu1d() {
        let x = Array1::from_vec(vec![1.0, -1.0, 0.0]);
        let y = ActivationFunctions::relu1d(x);
        assert_eq!(y, Array1::from_vec(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_relu2d() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 0.0, 2.0]).unwrap();
        let y = ActivationFunctions::relu2d(x);
        assert_eq!(
            y,
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap()
        );
    }

    #[test]
    fn test_relu_backward1d() {
        let x = Array1::from_vec(vec![1.0, -1.0, 0.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let z = ActivationFunctions::relu_backward1d(x, y);
        assert_eq!(z, Array1::from_vec(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_relu_backward2d() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 0.0, 2.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let z = ActivationFunctions::relu_backward2d(x, y);
        assert_eq!(
            z,
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn test_logsoftmax1d() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.0]);
        let y = ActivationFunctions::logsoftmax1d(x);
        let z = Array1::from_vec(vec![-2.4401897, -1.4401897, -0.4401897, -3.4401897] as Vec<f64>);

        assert!(y.iter().zip(z.iter()).fold(true, |acc, x| acc
            && crate::util::approximate_equal(*x.0, *x.1, None)),)
    }

    #[test]
    fn test_logsoftmax2d() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 0.0, 3.0, 0.0]).unwrap();
        let y = ActivationFunctions::logsoftmax2d(x);
        let z = Array2::from_shape_vec(
            (3, 2),
            vec![
                -1.31326169,
                -0.31326169,
                -0.04858735,
                -3.04858735,
                -0.04858735,
                -3.04858735,
            ] as Vec<f64>,
        )
        .unwrap();
        assert!(y.iter().zip(z.iter()).fold(true, |acc, x| acc
            && crate::util::approximate_equal(*x.0, *x.1, None)),)
    }

    #[test]
    fn test_logsoftmax_backward1d() {
        let x = Array1::from_vec(vec![0.0, -3.0, 1.0]);
        let y = Array1::from_vec(vec![0.0, -2.0, 0.0]);
        let t = ActivationFunctions::logsoftmax_backward1d(x, y);
        let z = Array1::from_vec(vec![0.53077585, -1.97357422, 1.44279836]);
        assert!(t.iter().zip(z.iter()).fold(true, |acc, x| acc
            && crate::util::approximate_equal(*x.0, *x.1, None)),);
    }

    #[test]
    fn test_logsoftmax_backward2d() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, -3.0, 1.0]).unwrap();
        let y = Array2::from_shape_vec((2, 3), vec![2.0, 1.0, -1.0, 0.0, -2.0, 0.0]).unwrap();
        let t = ActivationFunctions::logsoftmax_backward2d(x.clone(), y.clone());
        let z = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.51054305,
                -0.33048191,
                -1.1800611,
                0.53077585,
                -1.97357422,
                1.44279836,
            ] as Vec<f64>,
        )
        .unwrap();
        assert!(t.iter().zip(z.iter()).fold(true, |acc, x| acc
            && crate::util::approximate_equal(*x.0, *x.1, None)),)
    }
}
