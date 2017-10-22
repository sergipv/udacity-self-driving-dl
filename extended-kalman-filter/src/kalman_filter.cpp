#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}


void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();

  VectorXd y = z - H_ * x_; // H_ * x_ = z_pred
  UpdateWithStateDiff(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // x_ is in cartesian coordinates, and we need polar coordinates to compare
  // with z, which is encoded in polar coordinates.
  float range = sqrt(x_(0) * x_(0) + x_(1) *x_(1));
  float angle = atan(x_(1) / x_(0));
  float range_rate = (x_(0) * x_(2) + x_(1) * x_(3)) / range;

  VectorXd z_pred(3);
  z_pred << range, angle, range_rate;

  VectorXd y = z - z_pred;
  UpdateWithStateDiff(y);
}

void KalmanFilter::UpdateWithStateDiff(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}
