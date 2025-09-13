import React from 'react';
import { Header } from '../common/Header';
import { InfoComp } from './InfoComp';
import { LeftNavigation } from './LeftNavigation';

function SummaryPage() {
  return (
    <div>
      <Header></Header>
      <LeftNavigation></LeftNavigation>
      <InfoComp></InfoComp>
    </div>
  );
}

// comment

export default SummaryPage;
